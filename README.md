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
│   ├── oxideav-vorbis/       # Vorbis audio codec (decoder + encoder)
│   ├── oxideav-flac/         # FLAC native container + decoder + encoder
│   ├── oxideav-opus/         # Opus codec (header parsing; decoder TBD)
│   ├── oxideav-mkv/          # Matroska / WebM container (EBML), demux + mux
│   ├── oxideav-mp4/          # MP4 / ISO base media file format, demux + mux
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

Containers (probe / demux / mux): WAV, FLAC native, Ogg, Matroska, MP4
(demux + moov-at-end mux; fast-start is future work), IFF. Cross-container
remux works for any pair whose codecs don't require rewriting (FLAC ↔ MKV,
Ogg ↔ MKV, MP4 → FLAC/MKV, FLAC/PCM → MP4).

**Codecs**:

| Codec           | Decode                         | Encode                   |
|-----------------|--------------------------------|--------------------------|
| PCM (s8/16/24/32/f32) | ✅ all variants          | ✅ all variants          |
| FLAC            | ✅ bit-exact vs reference      | ✅ bit-exact vs reference |
| Vorbis          | ✅ matches lewton/ffmpeg        | ✅ real audio (tier 1)   |
| Opus            | header parsing only            | —                        |
| MOD (ProTracker)| ✅ 4-ch Paula mixer + effects  | —                        |
| 8SVX (Amiga IFF)| ✅                             | —                        |
| MP1/MP2/MP3     | header only (scaffold)         | —                        |
| AAC-LC          | header only (scaffold)         | —                        |
| CELT            | range decoder only (scaffold)  | —                        |
| Speex           | header parser (scaffold)       | —                        |
| GSM 06.10       | ✅ full RPE-LTP                | —                        |
| G.723.1 / G.728 / G.729 | scaffolds              | —                        |
| **MJPEG (video)** | ✅ baseline 4:2:0/4:2:2/4:4:4/grey | ✅ baseline 4:2:0/4:2:2/4:4:4 |
| **FFV1 (video)**  | ✅ self-roundtrip bit-exact (v3, 4:2:0 / 4:4:4) | ✅ (ffmpeg interop is follow-up) |
| **MPEG-1 video**  | 🔶 headers + pipeline wired; I-frame AC decode gap to close | — |

The Vorbis decoder passes bit-exact roundtrips against lewton and
matches ffmpeg's output within float rounding. The Vorbis encoder
emits recognisable audio today (sine-wave Goertzel ratio ~32× at
target freq vs noise) with rough quality; a multi-session effort to
reach libvorbis-quality parity is ongoing.

CLI verbs: `list`, `probe`, `remux`, `transcode`. Example:

```
$ oxideav transcode song.flac song.wav
Transcoded song.flac → song.wav (pcm_s16le): 482 pkts in, 482 frames decoded, 482 pkts out
```

## Roadmap

1. ✅ Workspace, core types, codec/container traits
2. ✅ `oxideav-basic`: WAV container + PCM codec
3. ✅ `oxideav` aggregator + CLI (`list`, `probe`, `remux`, `transcode`)
4. ✅ Source/sink pipeline with per-stream routing and copy-or-transcode decisions
5. ✅ Ogg container with byte-faithful page boundary preservation
6. ✅ FLAC native container + codec (decode + encode, both bit-exact)
7. ✅ Matroska demux + mux; MP4 demux + mux (moov-at-end)
8. ✅ Vorbis decoder + initial encoder
9. ✅ Amiga IFF + 8SVX + ProTracker MOD playback
10. Vorbis encoder — **in progress** toward libvorbis-quality parity
   (long+short blocks, stereo, wider residue VQ, then psy floor)
11. Opus decoder (SILK + CELT, RFC 6716) — major project
12. MP3 / AAC-LC full decoders (scaffolds today)
13. Filters: resample, sample-format conversion, pixel-format conversion, scale
14. ✅ First video codec: MJPEG (baseline JPEG decode + encode)
15. Video codecs: MPEG-1 / FFV1 / VP8 next

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
