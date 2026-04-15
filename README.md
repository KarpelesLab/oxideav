# oxideav

A **100% pure Rust** media transcoding and streaming framework. No C libraries, no FFI wrappers, no `*-sys` crates — just Rust, all the way down.

## Goals

- **Pure Rust implementation.** Never depend on `ffmpeg`, `libav`, `x264`, `libvpx`, `libopus`, or any other C library — directly or transitively. Every codec, container, and filter is implemented from the spec.
- **Clean abstractions** for codecs, containers, timestamps, and streaming formats.
- **Composable pipelines**: media input → demux → decode → transform → encode → mux → output, with pass-through mode for remuxing without re-encoding.
- **Modular workspace**: per-format crates for complex modern codecs/containers, a shared crate for simple standard formats, and an aggregator crate that ties them together behind Cargo features.

## Non-goals

- Wrapping existing C codec libraries.
- Perfect feature parity with FFmpeg on day one. Codec and container coverage grows incrementally.
- GPU-specific acceleration (may come later through pure-Rust compute libraries, but never C drivers).

## Workspace layout

```
oxideav/
├── crates/
│   ├── oxideav-core/         # primitives: Rational, Timestamp, Packet, Frame, formats
│   ├── oxideav-codec/        # codec traits: Encoder, Decoder, CodecId, registry glue
│   ├── oxideav-container/    # container traits: Demuxer, Muxer, registry glue
│   ├── oxideav-pipeline/     # pipeline composition (source → transforms → sink)
│   │
│   ├── oxideav-basic/        # simple / standard formats grouped together:
│   │                         #   PCM variants, WAV, Y4M (planned), …
│   │
│   ├── oxideav-ogg/          # Ogg container (RFC 3533): pages, packets, CRC32.
│   │                         #   Codec-agnostic transport layer.
│   ├── oxideav-vorbis/       # Vorbis audio codec (currently: id parser; decoder TBD)
│   ├── oxideav-flac/         # FLAC native container + decoder + encoder
│   ├── oxideav-opus/         # Opus codec (header parsing; decoder TBD)
│   ├── oxideav-mkv/          # Matroska / WebM container (EBML), demux + mux
│   ├── oxideav-mp4/          # MP4 / ISO base media file format, demux only
│   ├── oxideav-<format>/     # one crate per future complex format:
│   │                         #   oxideav-mp4, oxideav-h264, oxideav-av1, …
│   │
│   ├── oxideav/              # aggregator: re-exports + feature-gated registry.
│   │                         # Depend on this crate to get access to all codecs
│   │                         # and containers you enable via features.
│   │
│   └── oxideav-cli/          # `oxideav` command-line frontend (uses the aggregator)
└── Cargo.toml                # workspace manifest
```

### Why split formats into separate crates?

- **Complex codecs are large.** An H.264 or Opus implementation is tens of thousands of lines. Keeping each one in its own crate means users who don't need H.264 don't pay for it in build time, binary size, or audit scope.
- **Parallel compilation.** Independent crates compile concurrently.
- **Clean API boundaries.** Each format crate only depends on `oxideav-core`, `oxideav-codec`, and/or `oxideav-container` — never on other format crates. Cross-format glue lives in the aggregator.
- **Opt-in dependencies.** The aggregator crate uses Cargo features (`oxideav = { features = ["mkv", "opus"] }`) so downstream users pick exactly the formats they need.

### What goes in `oxideav-basic`?

Formats that are:
- Small (hundreds of lines, not thousands),
- Standard and stable (RFC-pinned, no algorithm variants to track),
- Useful as building blocks (PCM is needed any time you touch raw audio).

If a format grows beyond that — multiple profiles, complex bitstream parsing, optional tooling — it gets promoted to its own crate.

## Core concepts

- **Packet** — a chunk of compressed (encoded) data belonging to one stream, with timestamps.
- **Frame** — a chunk of uncompressed data (audio samples or a video picture).
- **Stream** — one media track inside a container (audio, video, subtitle…).
- **TimeBase / Timestamp** — rational time base per stream; timestamps are integers in that base.
- **Demuxer** — reads a container, emits Packets per stream.
- **Decoder** — turns Packets of a given codec into Frames.
- **Encoder** — turns Frames into Packets.
- **Muxer** — writes Packets into an output container.
- **Pipeline** — connects these pieces. A pipeline can pass Packets straight from Demuxer to Muxer (remux, no quality loss) or route through Decoder → [Filter] → Encoder.

## Current status

Probe / remux / **decode** / transcode end-to-end for FLAC and PCM/WAV. Pure-
Rust FLAC decoder verified bit-exact against the reference encoder for 16-bit
mono, 16-bit stereo with channel decorrelation, and 24-bit stereo (decoded
audio MD5 matches both ffmpeg and the original PCM the FLACs were encoded
from).

| Format        | Container                 | Codec                | Probe | Remux | Decode | Encode |
|---------------|---------------------------|----------------------|:-----:|:-----:|:------:|:------:|
| PCM in WAV    | `oxideav-basic` (wav)     | `oxideav-basic` (pcm)|  ✅   |  ✅   |   ✅   |   ✅   |
| FLAC native   | `oxideav-flac`            | `oxideav-flac`       |  ✅   |  ✅   |   ✅   |   ✅   |
| Ogg Vorbis    | `oxideav-ogg`             | `oxideav-vorbis`     |  ✅   |  ✅   |        |        |
| Ogg Opus      | `oxideav-ogg`             | `oxideav-opus`       |  ✅   |  ✅   |        |        |
| Matroska      | `oxideav-mkv` (FLAC/Opus/Vorbis/PCM) | (via codec crate) |  ✅   |  ✅   |   ✅\*  |        |
| MP4/M4A       | `oxideav-mp4` (FLAC/ALAC/AAC/Opus/H264/...) | (via codec crate) |  ✅   |  ✅\*\* |  ✅\*  |        |

\* When combined with a codec whose decoder is implemented (FLAC today).

\*\* MP4 demuxing is implemented; MP4 muxing is deferred (requires sample-table
buffering or a two-pass write).

Cross-container remux works: FLAC ↔ MKV, Ogg ↔ MKV, MP4 → FLAC/MKV with
decoded-audio MD5 preserved for FLAC and Vorbis. Opus→MKV produces a
playable stream with a minor end-trim discrepancy still being tracked.

CLI verbs: `list`, `probe`, `remux`, `transcode`. Example:

```
$ oxideav transcode song.flac song.wav
Transcoded song.flac → song.wav (pcm_s16le): 482 pkts in, 482 frames decoded, 482 pkts out
```

## Roadmap

1. ✅ Workspace, core types, codec/container traits
2. ✅ `oxideav-basic`: WAV container + PCM codec
3. ✅ `oxideav` aggregator + CLI (`list`, `probe`, `remux`, `transcode`)
4. ✅ Pipeline composition: passthrough remux + single-stream transcode
5. ✅ Ogg container with byte-faithful page boundary preservation
6. ✅ FLAC native container + codec (decode + encode, both bit-exact)
7. ✅ Vorbis + Opus identification headers (probe + remux today)
8. Vorbis decoder (codebooks, floors, residues, MDCT) — major project
9. Opus decoder (SILK + CELT, RFC 6716) — major project
10. Filters: resample, sample-format conversion, pixel-format conversion, scale
11. Multi-stream transcode + filter graph
12. Expand catalog: MP4/MKV containers, AAC, more video, …

## Building

```
cargo build --workspace
cargo test --workspace
```

The `oxideav` binary is produced by the `oxideav-cli` crate:

```
cargo run -p oxideav-cli -- --help
```

## License

TBD (likely MIT OR Apache-2.0).
