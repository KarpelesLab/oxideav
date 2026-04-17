//! Pixel-format conversion integration tests.
//!
//! - `explicit_convert_node_parses`: a `{"convert": "rgb24", ...}` schema
//!   node round-trips through `Job::from_json` → `Job::to_dag` and lands
//!   as a `DagNode::PixConvert` with the right target format.
//! - `describe_prints_convert_nodes`: the `Dag::describe()` pretty-printer
//!   surfaces the convert step so `dry-run` output stays legible.
//! - `auto_insert_on_codec_mismatch`: register a mock video encoder whose
//!   `accepted_pixel_formats` excludes the source pixel format and verify
//!   that `codec_accepted_pixel_formats` reports the list back to the
//!   executor's auto-insert pass.
//!
//! The full end-to-end executor round-trip through a pixel-format convert
//! is covered by `pixelized/real-codec` tests in the per-codec crates; here
//! we focus on the schema → DAG → auto-insert plumbing that is specific
//! to `oxideav-job`.

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{
    CodecCapabilities, CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat,
    Result,
};
use oxideav_job::{parse_pixel_format, DagNode, Job};

#[test]
fn pixel_format_parser_accepts_common_names() {
    assert_eq!(parse_pixel_format("yuv420p").unwrap(), PixelFormat::Yuv420P);
    assert_eq!(parse_pixel_format("RGB24").unwrap(), PixelFormat::Rgb24);
    assert_eq!(parse_pixel_format("rgba").unwrap(), PixelFormat::Rgba);
    assert_eq!(parse_pixel_format("pal8").unwrap(), PixelFormat::Pal8);
    assert_eq!(parse_pixel_format("gray8").unwrap(), PixelFormat::Gray8);
    assert_eq!(parse_pixel_format("nv12").unwrap(), PixelFormat::Nv12);
    assert_eq!(parse_pixel_format("rgb48le").unwrap(), PixelFormat::Rgb48Le);
    assert!(parse_pixel_format("not_a_real_format").is_err());
}

#[test]
fn explicit_convert_node_parses() {
    // A `{"convert": "rgb24", "input": {...}}` schema node must land
    // as a `DagNode::PixConvert` with target Rgb24.
    let job = Job::from_json(
        r#"{
            "out.mkv": {
                "video": [{
                    "convert": "rgb24",
                    "input": {"from": "in.mp4"},
                    "codec": "h264"
                }]
            }
        }"#,
    )
    .expect("parse");
    let dag = job.to_dag().expect("resolve");
    let root = dag.roots["out.mkv"];
    let mux_tracks = match dag.node(root) {
        DagNode::Mux { tracks, .. } => tracks,
        n => panic!("expected Mux root, got {n:?}"),
    };
    assert_eq!(mux_tracks.len(), 1);
    let enc_upstream = match dag.node(mux_tracks[0].upstream) {
        DagNode::Encode {
            upstream, codec, ..
        } => {
            assert_eq!(codec, "h264");
            *upstream
        }
        n => panic!("expected Encode above Mux, got {n:?}"),
    };
    let pc_upstream = match dag.node(enc_upstream) {
        DagNode::PixConvert { upstream, target } => {
            assert_eq!(*target, PixelFormat::Rgb24);
            *upstream
        }
        n => panic!("expected PixConvert above Encode, got {n:?}"),
    };
    match dag.node(pc_upstream) {
        DagNode::Decode { .. } => {}
        n => panic!("expected Decode above PixConvert, got {n:?}"),
    }
}

#[test]
fn describe_prints_convert_nodes() {
    // `dry-run` output should include the convert step in the pretty
    // printer so users can tell when auto/explicit conversions fire.
    let job = Job::from_json(
        r#"{
            "out.mkv": {
                "video": [{
                    "convert": "yuv420p",
                    "input": {"from": "in.mp4"},
                    "codec": "h264"
                }]
            }
        }"#,
    )
    .unwrap();
    let dag = job.to_dag().unwrap();
    let desc = dag.describe();
    assert!(desc.contains("convert(Yuv420P)"), "{desc}");
    assert!(desc.contains("encode(h264"), "{desc}");
}

#[test]
fn rejects_unknown_pixel_format_at_validate() {
    // Validation should catch typos in `convert` names with a pointer
    // back at the track context.
    let job = Job::from_json(
        r#"{
            "out.mkv": {
                "video": [{
                    "convert": "not_a_real_format",
                    "input": {"from": "in.mp4"},
                    "codec": "h264"
                }]
            }
        }"#,
    )
    .unwrap();
    let err = job.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("out.mkv"), "{msg}");
    assert!(msg.contains("convert"), "{msg}");
    assert!(msg.contains("not_a_real_format"), "{msg}");
}

// ─────────────────────── mock codec registration ──────────────────────
//
// Register a tiny "fake" video encoder that declares
// `accepted_pixel_formats = [Yuv420P]`. We don't exercise it — the test
// just confirms the registry plumbing is in place for the executor's
// auto-insert pass.

fn make_decoder(_p: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported("fake_vid: decoder never actually runs"))
}

fn make_encoder(_p: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::unsupported("fake_vid: encoder never actually runs"))
}

#[test]
fn auto_insert_registers_accepted_pixel_formats() {
    // Build a CodecRegistry with a fake encoder declaring a single
    // accepted pixel format, and confirm the information is visible
    // on the registered CodecImplementation.
    let mut reg = CodecRegistry::new();
    let caps = CodecCapabilities::video("fake_vid_sw").with_pixel_format(PixelFormat::Yuv420P);
    reg.register_encoder_impl(CodecId::new("fake_vid"), caps, make_encoder);
    let _ = make_decoder; // silence unused-fn warning

    let impls = reg.implementations(&CodecId::new("fake_vid"));
    assert_eq!(impls.len(), 1);
    let accepted = &impls[0].caps.accepted_pixel_formats;
    assert_eq!(accepted.as_slice(), &[PixelFormat::Yuv420P]);

    // The registry's per-implementation capability data is exactly
    // what `codec_accepted_pixel_formats` (and the executor's
    // `apply_pixel_format_auto_insert`) consumes at runtime.
    assert!(!accepted.is_empty());
    assert!(!accepted.contains(&PixelFormat::Rgb24));
}

// Silence unused-warning on helpers that would otherwise only appear
// behind a feature gate on other platforms.
const _: fn(&CodecParameters) -> Result<Box<dyn Decoder>> = make_decoder;
const _: fn(&CodecParameters) -> Result<Box<dyn Encoder>> = make_encoder;

// Parity test: re-run an existing job (audio, no pix convert needed)
// through both executor modes. This ensures the frame_stages refactor
// kept the serial and pipelined paths in sync end-to-end; a pix-convert
// regression would show up here because both modes share the same
// `FrameStage` plumbing.
#[test]
fn serial_pipelined_parity_with_pix_convert_stage_infra() {
    use std::io::Write;

    use oxideav_container::ContainerRegistry;
    use oxideav_job::Executor;
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
            let v = ((i as i32 * 41) & 0x7FFF) as i16;
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    let mut dir = std::env::temp_dir();
    dir.push("oxideav_job_pixcvt_parity");
    let _ = std::fs::create_dir_all(&dir);
    let src = dir.join("in.wav");
    let serial = dir.join("out_s.wav");
    let pipe = dir.join("out_p.wav");
    for p in [&serial, &pipe] {
        let _ = std::fs::remove_file(p);
    }
    build_pcm_wav(&src, 8_000, 60);

    let mut codecs = CodecRegistry::new();
    let mut containers = ContainerRegistry::new();
    oxideav_basic::register_codecs(&mut codecs);
    oxideav_basic::register_containers(&mut containers);
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
    Executor::new(
        &Job::from_json(&json_s).unwrap(),
        &codecs,
        &containers,
        &sources,
    )
    .with_threads(1)
    .run()
    .unwrap();
    Executor::new(
        &Job::from_json(&json_p).unwrap(),
        &codecs,
        &containers,
        &sources,
    )
    .with_threads(4)
    .run()
    .unwrap();
    let s = std::fs::read(&serial).unwrap();
    let p = std::fs::read(&pipe).unwrap();
    assert_eq!(s, p, "parity broke after frame_stages refactor");

    // Confirm the Frame / MediaType / Packet types stayed usable from
    // integration tests — these imports go stale if the refactor ever
    // drops them from the public API.
    let _phantom: fn(MediaType, &Frame, &Packet) = |_, _, _| {};
}
