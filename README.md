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

Container format detection is content-based: each container ships a
probe that scores the first 256 KB against its magic bytes. The file
extension is a tie-breaker hint, not the source of truth — a `.mp4`
that's actually a WAV opens correctly.

Containers (probe / demux / mux): WAV (LIST/INFO metadata), FLAC
native (VORBIS_COMMENT), Ogg (Vorbis/Opus/Theora comments + last-page
granule), Matroska (Info\Title + Tags), MP4 (udta + iTunes ilst +
mvhd; brand presets `mp4`/`mov`/`ismv`, optional `faststart`;
fragmented output is future work), AVI (LIST INFO + avih duration,
MJPEG/FFV1/PCM payloads), IFF (NAME/AUTH/ANNO/(c)/CHRS), MOD, S3M.
Cross-container remux works for any pair whose codecs don't require
rewriting (FLAC ↔ MKV, Ogg ↔ MKV, MP4 ↔ MOV, MP4 → FLAC/MKV,
FLAC/PCM → MP4, MJPEG ↔ AVI).

**Codecs**:

| Codec           | Decode                         | Encode                   |
|-----------------|--------------------------------|--------------------------|
| PCM (s8/16/24/32/f32) | ✅ all variants          | ✅ all variants          |
| FLAC            | ✅ bit-exact vs reference      | ✅ bit-exact vs reference |
| Vorbis          | ✅ matches lewton/ffmpeg        | ✅ stereo coupling + ATH floor; ffmpeg accepts; up to 14525× Goertzel |
| Opus            | TOC + framing + CELT frame-header bit-exact; CELT energy/PVQ/MDCT pending; SILK/Hybrid → Unsupported | — |
| MOD (ProTracker)| ✅ 4-ch Paula mixer + effects  | —                        |
| S3M (Scream Tracker 3) | ✅ 8/16-bit, A/B/C/D/E/F/G/H/J/K/L/O/Q/R/S8x/T/V/X effects | — |
| 8SVX (Amiga IFF)| ✅                             | —                        |
| MP1 / MP2       | header only (scaffold)         | —                        |
| MP3 (Layer III) | ✅ MPEG-1 LSF (Huffman 0-13/15-16/24 transcribed; 44/44 frames decode; 440 Hz Goertzel-dominates); intensity stereo + MPEG-2 LSF / 2.5 + CRC pending | — |
| AAC-LC          | ✅ mono + stereo; ICS info/section/scalefactor/spectrum + M/S + IMDCT 2048/256 + sine/KBD windows; mono Goertzel 144×, stereo 316×; SBR/PS/CCE/PCE/Main/SSR/LTP → Unsupported | — |
| CELT            | range decoder + frame-header decode; band-energy / PVQ / MDCT pending | — |
| Speex           | header parser (scaffold)       | —                        |
| GSM 06.10       | ✅ full RPE-LTP                | —                        |
| G.723.1 / G.728 / G.729 | scaffolds              | —                        |
| **MJPEG (video)** | ✅ baseline 4:2:0/4:2:2/4:4:4/grey | ✅ baseline 4:2:0/4:2:2/4:4:4 |
| **FFV1 (video)**  | ✅ self-roundtrip + ffmpeg→us (v3, 4:2:0 / 4:4:4) | ✅ (us→ffmpeg closes a 2-byte footer gap) |
| **MPEG-1 video**  | ✅ I+P+B frames (GOP decode, display-order reorder) | ✅ I-frames (round-trip 99.14% within ±8 LSB; ffmpeg accepts) |
| **MPEG-4 Part 2 / XVID / DivX** | ✅ I-VOP (PSNR 68.94 dB / 100% within 2 LSB on 64×64; resync markers honoured); P-VOP pending | — |
| **Theora (video)** | ✅ I + P frames 4:2:0 (100% match vs ffmpeg); 4:4:4 P-frames at 95.8% | — |

## Playback

An opt-in binary crate `oxideplay` implements a reference player with
SDL2 (audio + video) and a crossterm TUI. **The `oxideplay` crate is
the ONLY place in the workspace allowed to depend on SDL2 or any other
non-pure-Rust library** — the core `oxideav` library keeps its pure-
Rust invariant so it can ship on servers without graphical deps.

```
cargo run -p oxideplay -- /path/to/file.mkv
```

Keybinds: `q` quit, `space` pause, `← / →` seek ±10 s, `↑ / ↓` seek
±1 min (up = forward, down = back), `pgup / pgdn` seek ±10 min, `*`
volume up, `/` volume down. Works from the SDL window (when a video
stream is present) or from the TTY.

## CLI

`oxideav` command-line verbs: `list`, `probe`, `remux`, `transcode`. Example:

```
$ oxideav transcode song.flac song.wav
Transcoded song.flac → song.wav (pcm_s16le): 482 pkts in, 482 frames decoded, 482 pkts out
```

## Roadmap

1. ✅ Workspace, core types, codec/container traits
2. ✅ `oxideav-basic`: WAV container + PCM codec
3. ✅ `oxideav` aggregator + CLI (`list`, `probe`, `remux`, `transcode`)
4. ✅ Source/sink pipeline with per-stream routing and copy-or-transcode decisions
5. ✅ Content-based container probe (extension is a hint, not the source of truth)
6. ✅ Ogg container with byte-faithful page boundary preservation
7. ✅ FLAC native container + codec (decode + encode, both bit-exact)
8. ✅ Matroska demux + mux; MP4 demux + mux (moov-at-end + faststart)
9. ✅ AVI demux + mux (MJPEG / FFV1 / PCM payloads)
10. ✅ Amiga IFF + 8SVX + ProTracker MOD + Scream Tracker 3 (S3M) playback
11. ✅ Vorbis decoder + encoder (stereo coupling + ATH floor; ffmpeg accepts)
12. ✅ MJPEG, FFV1, MPEG-1 video, MPEG-4 Part 2, Theora I-frame video decoders
13. ✅ MJPEG, FFV1, MPEG-1 video I-frame encoders
14. ✅ MP3 (Layer III) + AAC-LC decoders
15. Opus decoder — CELT energy/PVQ/MDCT next, then SILK + Hybrid (RFC 6716)
16. MPEG-4 Part 2 P-VOPs + Theora encoder + AAC-LC encoder + MP3 encoder
17. Filters: resample, sample-format conversion, pixel-format conversion, scale
18. Video codecs: VP8 / VP9 / AV1 — much later

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

MIT — see [`LICENSE`](LICENSE). Copyright © 2026 Karpelès Lab Inc.
