//! Parity + safety tests for the pipelined executor.
//!
//! - `serial_pipelined_parity`: the same job run with `threads=1` and
//!   `threads=4` must produce byte-identical output (copy path) and
//!   consistent stats.
//! - `abort_propagates_from_sink`: a sink that returns an error after
//!   the first write must cause the executor to return that error and
//!   join every worker without deadlocking.
//! - `transcode_parity`: same as above but exercises decoder → encoder
//!   threads (not just copy), so races in the decode/encode boundary
//!   would show up.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::{Error, Frame, MediaType, Packet, Result, StreamInfo};
use oxideav_job::{Executor, Job, JobSink};
use oxideav_source::SourceRegistry;

fn build_pcm_wav(path: &std::path::Path, sample_rate: u32, ms: u32) {
    let n_samples = (sample_rate as u64 * ms as u64 / 1000) as u32;
    let byte_rate = sample_rate * 2;
    let data_sz = n_samples * 2;
    let riff_sz = 36 + data_sz;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"RIFF").unwrap();
    f.write_all(&riff_sz.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_sz.to_le_bytes()).unwrap();
    for i in 0..n_samples {
        // Non-silent, non-trivial content so bytes differ between runs.
        let v = ((i as i32 * 73) & 0x7FFF) as i16;
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn registries() -> (CodecRegistry, ContainerRegistry) {
    let mut c = CodecRegistry::new();
    let mut co = ContainerRegistry::new();
    oxideav_basic::register_codecs(&mut c);
    oxideav_basic::register_containers(&mut co);
    (c, co)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("oxideav_job_pipe_{name}"));
    let _ = std::fs::create_dir_all(&p);
    p
}

fn run_with_threads(
    json: &str,
    threads: usize,
    codecs: &CodecRegistry,
    containers: &ContainerRegistry,
    sources: &SourceRegistry,
) {
    let job = Job::from_json(json).expect("parse");
    Executor::new(&job, codecs, containers, sources)
        .with_threads(threads)
        .run()
        .expect("run");
}

#[test]
fn serial_pipelined_parity_copy_path() {
    let dir = tmp("parity_copy");
    let src = dir.join("in.wav");
    let serial = dir.join("out_serial.wav");
    let pipe = dir.join("out_pipe.wav");
    for p in [&serial, &pipe] {
        let _ = std::fs::remove_file(p);
    }
    build_pcm_wav(&src, 8_000, 100);

    let (codecs, containers) = registries();
    let sources = SourceRegistry::with_defaults();
    let escape = |p: &std::path::Path| p.display().to_string().replace('\\', "\\\\");

    let json_s = format!(
        r#"{{ "{}": {{ "audio": [{{"from": "{}"}}] }} }}"#,
        escape(&serial),
        escape(&src),
    );
    let json_p = format!(
        r#"{{ "{}": {{ "audio": [{{"from": "{}"}}] }} }}"#,
        escape(&pipe),
        escape(&src),
    );

    run_with_threads(&json_s, 1, &codecs, &containers, &sources);
    run_with_threads(&json_p, 4, &codecs, &containers, &sources);

    let s_bytes = std::fs::read(&serial).unwrap();
    let p_bytes = std::fs::read(&pipe).unwrap();
    assert_eq!(
        s_bytes,
        p_bytes,
        "copy-path byte mismatch between serial and pipelined ({} vs {} bytes)",
        s_bytes.len(),
        p_bytes.len()
    );
}

#[test]
fn serial_pipelined_parity_transcode() {
    let dir = tmp("parity_transcode");
    let src = dir.join("in.wav");
    let serial = dir.join("out_serial.wav");
    let pipe = dir.join("out_pipe.wav");
    for p in [&serial, &pipe] {
        let _ = std::fs::remove_file(p);
    }
    build_pcm_wav(&src, 8_000, 100);

    let (codecs, containers) = registries();
    let sources = SourceRegistry::with_defaults();
    let escape = |p: &std::path::Path| p.display().to_string().replace('\\', "\\\\");

    let json_s = format!(
        r#"{{ "{}": {{ "audio": [{{"from": "{}", "codec": "pcm_s16le"}}] }} }}"#,
        escape(&serial),
        escape(&src),
    );
    let json_p = format!(
        r#"{{ "{}": {{ "audio": [{{"from": "{}", "codec": "pcm_s16le"}}] }} }}"#,
        escape(&pipe),
        escape(&src),
    );
    run_with_threads(&json_s, 1, &codecs, &containers, &sources);
    run_with_threads(&json_p, 4, &codecs, &containers, &sources);

    let s_bytes = std::fs::read(&serial).unwrap();
    let p_bytes = std::fs::read(&pipe).unwrap();
    assert_eq!(
        s_bytes, p_bytes,
        "transcode byte mismatch between serial and pipelined"
    );
}

#[test]
fn abort_propagates_from_sink() {
    // A sink that errors on the first packet write. With the pipelined
    // executor, this must (a) surface the injected error from `run()`,
    // and (b) join every worker thread without deadlocking.
    struct FailingSink {
        called: Arc<AtomicBool>,
    }
    impl JobSink for FailingSink {
        fn start(&mut self, _streams: &[StreamInfo]) -> Result<()> {
            Ok(())
        }
        fn write_packet(&mut self, _kind: MediaType, _pkt: &Packet) -> Result<()> {
            self.called.store(true, Ordering::SeqCst);
            Err(Error::other("sink error injected by test"))
        }
        fn write_frame(&mut self, _kind: MediaType, _f: &Frame) -> Result<()> {
            Err(Error::other("unexpected frame path"))
        }
        fn finish(&mut self) -> Result<()> {
            Ok(())
        }
    }

    let dir = tmp("abort");
    let src = dir.join("in.wav");
    build_pcm_wav(&src, 8_000, 200);

    let (codecs, containers) = registries();
    let sources = SourceRegistry::with_defaults();
    let called = Arc::new(AtomicBool::new(false));

    // @null is a reserved sink so the executor lets us swap in our own
    // JobSink via with_sink_override. A non-reserved alias like `@sink`
    // would be mistaken for an intermediate.
    let escape = |p: &std::path::Path| p.display().to_string().replace('\\', "\\\\");
    let json = format!(
        r#"{{ "@null": {{ "audio": [{{"from": "{}"}}] }} }}"#,
        escape(&src),
    );

    let job = Job::from_json(&json).expect("parse");
    let err = Executor::new(&job, &codecs, &containers, &sources)
        .with_sink_override(
            "@null",
            Box::new(FailingSink {
                called: called.clone(),
            }),
        )
        .with_threads(4)
        .run()
        .expect_err("expected sink error to propagate");
    assert!(
        format!("{err}").contains("sink error injected"),
        "got: {err}"
    );
    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn pipelined_handles_cycle_error() {
    // Malformed job (cycle) must still return an error from run() under
    // the pipelined path — the validator runs first, before we spawn any
    // threads.
    let job = Job::from_json(
        r#"{
            "@a": {"all": [{"from": "@b"}]},
            "@b": {"all": [{"from": "@a"}]},
            "out.wav": {"audio": [{"from": "@a"}]}
        }"#,
    )
    .unwrap();
    let (codecs, containers) = registries();
    let sources = SourceRegistry::with_defaults();
    let err = Executor::new(&job, &codecs, &containers, &sources)
        .with_threads(4)
        .run()
        .expect_err("cycle");
    assert!(format!("{err}").contains("cycle"), "got: {err}");
}
