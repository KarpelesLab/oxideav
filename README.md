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
| Opus            | TOC + framing + CELT frame-header bit-exact + §4.3.2 coarse/fine energy; PVQ / IMDCT / post-filter pending; SILK / Hybrid → Unsupported | — |
| MOD (ProTracker)| ✅ 4-ch Paula mixer + effects  | —                        |
| S3M (Scream Tracker 3) | ✅ 8/16-bit, A/B/C/D/E/F/G/H/J/K/L/O/Q/R/S8x/T/V/X effects | — |
| 8SVX (Amiga IFF)| ✅                             | —                        |
| MP1 / MP2       | header only (scaffold)         | —                        |
| MP3 (Layer III) | 🔶 MPEG-1 LSF decode runs end-to-end on real CBR clips (recognisable but pitch off ~5%; numerical-bisection vs puremp3 pending); intensity stereo + MPEG-2 LSF / 2.5 + CRC pending | ✅ CBR mono + stereo (44.1k / 48k @ 128/192 kbps); ffmpeg accepts; Goertzel 8523-57639× |
| AAC-LC          | ✅ mono + stereo; ICS info/section/scalefactor/spectrum + M/S + IMDCT 2048/256 + sine/KBD windows; mono Goertzel 144×, stereo 316×; SBR/PS/CCE/PCE/Main/SSR/LTP → Unsupported | ✅ mono + stereo (44.1k/48k); ffmpeg accepts; Goertzel 391-632× |
| CELT            | range decoder + frame-header decode + coarse/fine band energy; PVQ / IMDCT / post-filter pending | — |
| Speex           | ✅ NB CELP sub-modes 1..=8 (Goertzel 6.76e7 at 24 kbps); WB / postfilter / intensity stereo → Unsupported | — |
| GSM 06.10       | ✅ full RPE-LTP                | —                        |
| G.723.1 / G.728 / G.729 | scaffolds              | —                        |
| **MJPEG (video)** | ✅ baseline 4:2:0/4:2:2/4:4:4/grey | ✅ baseline 4:2:0/4:2:2/4:4:4 |
| **FFV1 (video)**  | ✅ self-roundtrip + ffmpeg→us (v3, 4:2:0 / 4:4:4) | ✅ (us→ffmpeg closes a 2-byte footer gap) |
| **H.263 (video)** | ✅ I-pictures (100% within 2 LSB vs ffmpeg on sub-QCIF / QCIF / CIF, q=5 + q=15); P-pictures + Annexes D/E/F/G/I/J/T pending | — |
| **MPEG-1 video**  | ✅ I+P+B frames (GOP decode, display-order reorder) | ✅ I + P frames (P-frame round-trip 95.94% within ±16 LSB, ~2.12× compression vs I-only at GOP=3) |
| **MPEG-4 Part 2 / XVID / DivX** | ✅ I-VOP + P-VOP at half-pel MC (PSNR 67-69 dB / 100% within 2 LSB on 64×64 GOP); quarter-pel pending | — |
| **H.264 (video)** | NAL + SPS + PPS + slice header parse; baseline I-slice CAVLC + intra prediction + IDCT pending; CABAC / P/B / MBAFF → Unsupported | — |
| **H.265 (HEVC, video)** | NAL + VPS + SPS + PPS + slice segment header parse; CTU decode (CABAC + transforms + SAO + ALF) → Unsupported | — |
| **Theora (video)** | ✅ I + P frames 4:2:0 (100% match vs ffmpeg); 4:4:4 P-frames at 95.8% | ✅ I-frames (99.45% match round-trip + ffmpeg interop, libtheora-default setup header) |
| **VP8 (video)** | 🔶 Full RFC 6386 pipeline (bool decoder, header, intra, tokens, IDCT, loop filter, IVF container); MB(0,0) bit-perfect, neighbour-context propagation bug pending | — |
| **VP9 (video)** | Uncompressed + partial compressed header parse; tile decode → Unsupported | — |
| **AV1 (video)** | OBU + sequence header + frame header parse; tile decode → Unsupported | — |

## Playback

An opt-in binary crate `oxideplay` implements a reference player with
SDL2 (audio + video) and a crossterm TUI. SDL2 is loaded **at runtime
via `libloading`** — `oxideplay` doesn't link against SDL2 at build
time, so the binary builds and ships without requiring SDL2 dev
headers. If SDL2 isn't installed on the target machine, the player
exits cleanly with a "library not found" message instead of failing
to start. The core `oxideav` library remains 100% pure Rust.

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

Done:

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
11. ✅ Vorbis decoder + encoder; AAC-LC decoder + encoder; MP3 decoder (partial) + CBR encoder
12. ✅ Speex narrowband CELP decoder; GSM 06.10 decoder
13. ✅ MJPEG, FFV1, MPEG-1 video (I+P+B), MPEG-4 Part 2 (I+P-VOP), Theora (I+P), H.263 (I) — full decoders
14. ✅ MJPEG, FFV1 (interop fix pending), MPEG-1 video (I+P), Theora (I) — encoders
15. ✅ H.264, H.265, VP9, AV1 — bitstream/header parsers ready for decode follow-ups
16. ✅ VP8 — full pipeline coded (neighbour-context bug pending)
17. ✅ `oxideplay` reference player with runtime-loaded SDL2 (libloading)

Next:

18. Opus — CELT PVQ + IMDCT + post-filter; then SILK + Hybrid (RFC 6716)
19. MP3 — bisect residual amplitude/pitch bug vs puremp3; intensity stereo; MPEG-2 LSF
20. VP8 — finish neighbour-context propagation; P-frame decode
21. H.264 — baseline I-slice decode (CAVLC + intra prediction + IDCT + deblocking)
22. H.265 / VP9 / AV1 — decode pipelines (multi-session per codec)
23. Encoder gaps: MPEG-4 Part 2, Theora P-frames, H.263, AAC short blocks
24. Filters: resample, sample-format conversion, pixel-format conversion, scale

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
