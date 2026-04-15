//! `oxideav` command-line frontend.

use clap::{Parser, Subcommand};
use oxideav::container::ReadSeek;
use oxideav::core::Error;
use oxideav::Registries;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "oxideav",
    version,
    about = "Pure-Rust media transcoding framework",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List compiled-in codecs and containers.
    List,
    /// Probe a media file and print stream information.
    Probe {
        /// Path to the input media file.
        input: PathBuf,
    },
    /// Remux an input file to a new container (no re-encoding).
    ///
    /// Only stream copy is supported for now; both sides must use the same codec.
    Remux {
        input: PathBuf,
        output: PathBuf,
        /// Override the output container format. Defaults to file extension.
        #[arg(long)]
        format: Option<String>,
    },
    /// Decode an input file and re-encode to a new codec.
    ///
    /// Today this is single-stream only. The output codec defaults to a PCM
    /// variant matching the decoded sample format (e.g. FLAC 16-bit → pcm_s16le).
    Transcode {
        input: PathBuf,
        output: PathBuf,
        /// Override the output codec id (e.g. "pcm_s16le", "pcm_f32le").
        #[arg(long)]
        codec: Option<String>,
        /// Override the output container format. Defaults to file extension.
        #[arg(long)]
        format: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let registries = Registries::with_all_features();

    let result = match cli.command {
        Command::List => cmd_list(&registries),
        Command::Probe { input } => cmd_probe(&registries, &input),
        Command::Remux {
            input,
            output,
            format,
        } => cmd_remux(&registries, &input, &output, format.as_deref()),
        Command::Transcode {
            input,
            output,
            codec,
            format,
        } => cmd_transcode(
            &registries,
            &input,
            &output,
            codec.as_deref(),
            format.as_deref(),
        ),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("oxideav: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_list(reg: &Registries) -> oxideav::core::Result<()> {
    let mut containers: Vec<_> = reg.containers.demuxer_names().collect();
    containers.sort_unstable();
    println!("Containers (demux):");
    for c in &containers {
        println!("  {c}");
    }

    let mut muxers: Vec<_> = reg.containers.muxer_names().collect();
    muxers.sort_unstable();
    println!("Containers (mux):");
    for c in &muxers {
        println!("  {c}");
    }

    println!();
    println!("Codecs:");
    println!(" D..... = Decoding supported");
    println!(" .E.... = Encoding supported");
    println!(" ..V... = Video codec");
    println!(" ..A... = Audio codec");
    println!(" ..S... = Subtitle codec");
    println!(" ..D... = Data codec");
    println!(" ..T... = Attachment codec");
    println!(" ...I.. = Intra frame-only codec");
    println!(" ....L. = Lossy compression");
    println!(" .....S = Lossless compression");
    println!(" ------");
    let mut rows: Vec<(String, String, String, bool)> = reg
        .codecs
        .all_implementations()
        .map(|(id, im)| {
            (
                im.caps.flag_string(),
                id.as_str().to_owned(),
                im.caps.implementation.clone(),
                im.caps.hardware_accelerated,
            )
        })
        .collect();
    rows.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
    for (flags, id, implementation, hw) in rows {
        let hw_tag = if hw { "  [HW]" } else { "" };
        println!("  {flags}  {id:<14}  ({implementation}){hw_tag}");
    }
    Ok(())
}

fn cmd_probe(reg: &Registries, input: &Path) -> oxideav::core::Result<()> {
    let format = format_for_path(reg, input)?;
    let file: Box<dyn ReadSeek> = Box::new(File::open(input)?);
    let demuxer = reg.containers.open_demuxer(&format, file)?;
    println!(
        "Input: {}\nFormat: {}",
        input.display(),
        demuxer.format_name()
    );
    for s in demuxer.streams() {
        let p = &s.params;
        print!(
            "  Stream #{} [{:?}]  codec={}  time_base={}",
            s.index,
            p.media_type,
            p.codec_id,
            s.time_base.as_rational()
        );
        if let (Some(ch), Some(sr)) = (p.channels, p.sample_rate) {
            print!("  audio {}ch @ {} Hz", ch, sr);
            if let Some(fmt) = p.sample_format {
                print!("  [{:?}]", fmt);
            }
        }
        if let (Some(w), Some(h)) = (p.width, p.height) {
            print!("  video {}x{}", w, h);
        }
        if let Some(br) = p.bit_rate {
            print!("  {} bps", br);
        }
        if let Some(d) = s.duration {
            print!("  duration={} ticks ({:.3}s)", d, s.time_base.seconds_of(d));
        }
        println!();
    }
    Ok(())
}

fn cmd_remux(
    reg: &Registries,
    input: &Path,
    output: &Path,
    format_override: Option<&str>,
) -> oxideav::core::Result<()> {
    let in_format = format_for_path(reg, input)?;
    let out_format = match format_override {
        Some(f) => f.to_owned(),
        None => format_for_path(reg, output)?,
    };

    let fin: Box<dyn ReadSeek> = Box::new(File::open(input)?);
    let mut demuxer = reg.containers.open_demuxer(&in_format, fin)?;

    let fout: Box<dyn oxideav::container::WriteSeek> = Box::new(File::create(output)?);
    let mut muxer = reg
        .containers
        .open_muxer(&out_format, fout, demuxer.streams())?;

    let n = oxideav::pipeline::remux(&mut *demuxer, &mut *muxer)?;
    println!(
        "Remuxed {} packet(s) from {} ({}) → {} ({})",
        n,
        input.display(),
        in_format,
        output.display(),
        out_format,
    );
    Ok(())
}

fn cmd_transcode(
    reg: &Registries,
    input: &Path,
    output: &Path,
    codec_override: Option<&str>,
    format_override: Option<&str>,
) -> oxideav::core::Result<()> {
    use oxideav::core::SampleFormat;
    use oxideav::pipeline::{transcode_simple, StreamPlan};

    let in_format = format_for_path(reg, input)?;
    let out_format = match format_override {
        Some(f) => f.to_owned(),
        None => format_for_path(reg, output)?,
    };
    let fin: Box<dyn ReadSeek> = Box::new(File::open(input)?);
    let mut demuxer = reg.containers.open_demuxer(&in_format, fin)?;

    // Pick an output codec. If user supplied one, use it. Otherwise pick a
    // PCM variant that matches the input stream's natural bit depth.
    let codec = match codec_override {
        Some(c) => c.to_owned(),
        None => {
            let in_streams = demuxer.streams();
            let stream = in_streams
                .first()
                .ok_or_else(|| oxideav::core::Error::invalid("no streams"))?;
            let fmt = stream.params.sample_format.unwrap_or(SampleFormat::S16);
            match fmt {
                SampleFormat::U8 => "pcm_u8",
                SampleFormat::S16 => "pcm_s16le",
                SampleFormat::S24 => "pcm_s24le",
                SampleFormat::S32 => "pcm_s32le",
                SampleFormat::F32 => "pcm_f32le",
                SampleFormat::F64 => "pcm_f64le",
                _ => "pcm_s16le",
            }
            .to_owned()
        }
    };

    let plan = StreamPlan::Reencode {
        output_codec: codec.clone(),
    };

    let fout: Box<dyn oxideav::container::WriteSeek> = Box::new(File::create(output)?);
    let registries_containers = &reg.containers;
    let out_format_owned = out_format.clone();
    let muxer_open = move |streams: &[oxideav::core::StreamInfo]| {
        registries_containers.open_muxer(&out_format_owned, fout, streams)
    };

    let stats = transcode_simple(&mut *demuxer, muxer_open, &reg.codecs, &plan)?;
    println!(
        "Transcoded {} → {} ({}): {} pkts in, {} frames decoded, {} pkts out",
        input.display(),
        output.display(),
        codec,
        stats.packets_in,
        stats.frames_decoded,
        stats.packets_out,
    );
    Ok(())
}

fn format_for_path(reg: &Registries, path: &Path) -> oxideav::core::Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| Error::FormatNotFound(format!("no extension on {}", path.display())))?;
    reg.containers
        .container_for_extension(ext)
        .map(|s| s.to_owned())
        .ok_or_else(|| Error::FormatNotFound(format!("no container registered for .{ext}")))
}
