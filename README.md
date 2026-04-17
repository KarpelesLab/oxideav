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
│   ├── oxideav-core/         # primitives: Rational, Timestamp, Packet, Frame, formats, ExecutionContext
│   ├── oxideav-codec/        # codec traits: Encoder, Decoder, CodecId, registry glue
│   ├── oxideav-container/    # container traits: Demuxer, Muxer, registry glue
│   ├── oxideav-pipeline/     # pipeline composition (source → transforms → sink)
│   ├── oxideav-source/       # generic I/O: SourceRegistry, file:// driver, BufferedSource
│   ├── oxideav-http/         # HTTP/HTTPS source via ureq + rustls (Range requests)
│   ├── oxideav-audio-filter/ # audio processing: Volume, NoiseGate, Echo, Resample, Spectrogram
│   ├── oxideav-pixfmt/       # pixel format conversion: RGB/YUV/Gray/Pal8, dither, palette gen
│   ├── oxideav-job/          # JSON transcode job graph + pipelined multithreaded executor
│   │
│   ├── oxideav-basic/        # simple / standard formats: PCM variants, WAV
│   │
│   ├── oxideav-ogg/          # Ogg container (RFC 3533)
│   ├── oxideav-vorbis/       # Vorbis audio (decoder + encoder)
│   ├── oxideav-opus/         # Opus audio (SILK NB/MB/WB 10+20ms + CELT decode, stereo CELT)
│   ├── oxideav-flac/         # FLAC native container + codec (decode + encode)
│   ├── oxideav-mkv/          # Matroska / WebM container (EBML), demux + mux
│   ├── oxideav-mp4/          # MP4 / ISO BMFF, demux + mux
│   ├── oxideav-avi/          # AVI container, demux + mux
│   ├── oxideav-iff/          # Amiga IFF / 8SVX
│   ├── oxideav-mod/          # ProTracker MOD player
│   ├── oxideav-s3m/          # Scream Tracker 3 player (stereo + SCx/SDx/SBx)
│   ├── oxideav-amv/          # AMV container + video + IMA-ADPCM audio (decode + encode)
│   ├── oxideav-webp/         # WebP image (VP8 lossy + VP8L lossless + animation)
│   ├── oxideav-png/          # PNG + APNG decoder + encoder
│   ├── oxideav-gif/          # GIF decoder + encoder (LZW, animation)
│   │
│   ├── oxideav-mp1/          # MPEG-1 Audio Layer I decoder
│   ├── oxideav-mp2/          # MPEG-1 Audio Layer II decoder + encoder
│   ├── oxideav-mp3/          # MP3 decoder + encoder
│   ├── oxideav-aac/          # AAC-LC decoder + encoder
│   ├── oxideav-celt/         # CELT standalone decoder
│   ├── oxideav-speex/        # Speex decoder (NB + WB, with formant postfilter)
│   ├── oxideav-gsm/          # GSM 06.10 decoder + encoder
│   ├── oxideav-g7231/        # G.723.1 scaffold
│   ├── oxideav-g728/         # G.728 scaffold
│   ├── oxideav-g729/         # G.729 scaffold
│   │
│   ├── oxideav-mjpeg/        # MJPEG decoder + encoder + still-JPEG container
│   ├── oxideav-ffv1/         # FFV1 v3 decoder + encoder
│   ├── oxideav-mpeg1video/   # MPEG-1 video decoder + encoder
│   ├── oxideav-mpeg4video/   # MPEG-4 Part 2 decoder + encoder
│   ├── oxideav-theora/       # Theora decoder + encoder
│   ├── oxideav-h263/         # H.263 decoder + encoder
│   ├── oxideav-h264/         # H.264 decoder (baseline I-slice skeleton)
│   ├── oxideav-h265/         # H.265 header parser
│   ├── oxideav-vp8/          # VP8 decoder (I + P frames) + IVF container
│   ├── oxideav-vp9/          # VP9 header parser
│   ├── oxideav-av1/          # AV1 header parser
│   ├── oxideav-prores/       # Apple ProRes scaffold (decoder/encoder not yet implemented)
│   ├── oxideav-jpegxl/       # JPEG XL scaffold (decoder/encoder not yet implemented)
│   ├── oxideav-jpeg2000/     # JPEG 2000 scaffold (decoder/encoder not yet implemented)
│   ├── oxideav-avif/         # AVIF scaffold (decoder/encoder not yet implemented)
│   │
│   ├── oxideav/              # aggregator: re-exports + feature-gated registry
│   ├── oxideav-cli/          # `oxideav` command-line frontend (list/probe/remux/transcode/run/validate/dry-run)
│   └── oxideplay/            # reference media player (SDL2 + crossterm TUI)
└── Cargo.toml                # workspace manifest
```

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

### Containers

Container format detection is content-based: each container ships a
probe that scores the first 256 KB against its magic bytes. The file
extension is a tie-breaker hint, not the source of truth — a `.mp4`
that's actually a WAV opens correctly.

| Container | Probe | Demux | Mux | Notes |
|-----------|:-----:|:-----:|:---:|-------|
| WAV       | ✅ | ✅ | ✅ | LIST/INFO metadata |
| FLAC      | ✅ | ✅ | ✅ | VORBIS_COMMENT, streaminfo |
| Ogg       | ✅ | ✅ | ✅ | Vorbis/Opus/Theora/Speex pages + comments |
| Matroska  | ✅ | ✅ | ✅ | MKV/MKA/MKS; DocType-aware probe |
| WebM      | ✅ | ✅ | ✅ | First-class: separate fourcc, codec whitelist (VP8/VP9/AV1/Vorbis/Opus) |
| MP4       | ✅ | ✅ | ✅ | mp4/mov/ismv brands, faststart, iTunes ilst metadata |
| AVI       | ✅ | ✅ | ✅ | LIST INFO, avih duration |
| IFF / 8SVX| ✅ | ✅ | — | Amiga IFF with NAME/AUTH/ANNO/(c)/CHRS |
| MOD       | ✅ | ✅ | — | ProTracker 4-channel modules |
| S3M       | ✅ | ✅ | — | Scream Tracker 3 modules |
| MP3       | ✅ | ✅ | ✅ | ID3v2, Xing/VBRI, frame sync |
| IVF       | ✅ | ✅ | — | VP8 elementary stream container |
| AMV       | ✅ | ✅ | — | Chinese MP4 player format (RIFF-like) |
| WebP      | ✅ | ✅ | — | RIFF/WEBP (lossy + lossless + animation) |
| PNG / APNG| ✅ | ✅ | ✅ | 8 + 16-bit, all color types, APNG animation |
| GIF       | ✅ | ✅ | ✅ | GIF87a/GIF89a, LZW, animation + NETSCAPE2.0 loop |
| JPEG      | ✅ | ✅ | ✅ | Still-image wrapper around the MJPEG codec |

Cross-container remux works for any pair whose codecs don't require
rewriting (FLAC ↔ MKV, Ogg ↔ MKV, MP4 ↔ MOV, etc.).

### Codecs

| Codec | Decode | Encode |
|-------|--------|--------|
| **PCM** (s8/16/24/32/f32) | ✅ all variants | ✅ all variants |
| **FLAC** | ✅ bit-exact vs reference | ✅ bit-exact vs reference |
| **Vorbis** | ✅ matches lewton/ffmpeg (type-0/1/2 residue) | ✅ stereo coupling + ATH floor |
| **Opus** | ✅ CELT mono+stereo; SILK NB/MB/WB mono 10+20 ms | — |
| **MP1** | ✅ all modes, RMS 2.9e-5 vs ffmpeg | — |
| **MP2** | ✅ all modes, RMS 2.9e-5 vs ffmpeg | ✅ CBR mono+stereo (greedy allocator, ~31 dB PSNR) |
| **MP3** | ✅ MPEG-1 Layer III (M/S stereo) | ✅ CBR mono+stereo |
| **AAC-LC** | ✅ mono+stereo, M/S, IMDCT | ✅ mono+stereo, ffmpeg accepts |
| **CELT** | ✅ full §4.3 pipeline (energy + PVQ + IMDCT + post-filter) | — |
| **Speex** | ✅ NB modes 1-8 + WB via QMF+SB-CELP (+ formant postfilter) | — |
| **GSM 06.10** | ✅ full RPE-LTP | ✅ full RPE-LTP (standard + WAV-49) |
| **G.723.1 / G.728 / G.729** | scaffold | — |
| **MJPEG** | ✅ baseline 4:2:0/4:2:2/4:4:4/grey | ✅ baseline |
| **FFV1** | ✅ v3, 4:2:0/4:4:4 | ✅ v3 |
| **MPEG-1 video** | ✅ I+P+B frames | ✅ I+P frames (half-pel ME, 42 dB PSNR) |
| **MPEG-4 Part 2** | ✅ I+P-VOP, half-pel MC | ✅ I+P-VOP (41-43 dB PSNR, 21% vs all-I) |
| **Theora** | ✅ I+P frames | ✅ I+P frames (45 dB PSNR, 3.7× vs all-I) |
| **H.263** | ✅ I+P pictures, half-pel MC | ✅ I+P pictures (100% bit-exact vs ffmpeg) |
| **H.264** | Baseline I-slice skeleton: CAVLC + intra-pred + transforms + deblocking; 100% on solid-gray IDR | — |
| **H.265 (HEVC)** | NAL + VPS/SPS/PPS/slice parse | — |
| **VP8** | ✅ I+P frames (6-tap sub-pel + MV decode + ref management) | — |
| **VP9** | Uncompressed + partial compressed header | — |
| **AV1** | OBU + sequence/frame header parse | — |
| **WebP VP8L** | ✅ full lossless (Huffman + LZ77 + transforms) | — |
| **AMV video** | ✅ (synthesised JPEG header + vertical flip) | — |
| **IMA-ADPCM (AMV)** | ✅ | ✅ (33.8 dB PSNR roundtrip) |
| **PNG / APNG** | ✅ 5 color types × 8/16-bit, all 5 filters, APNG animation | ✅ same matrix + APNG emit |
| **GIF** | ✅ GIF87a/89a, LZW, interlaced, animation | ✅ GIF89a, animation, per-frame palettes |
| **ProRes / JPEG XL / JPEG 2000 / AVIF** | scaffold (returns Unsupported) | scaffold |
| **MOD / S3M** | ✅ stereo + SCx/SDx/SBx effects on S3M (player, no encoder planned) | — |
| **8SVX** | ✅ | — |

### Audio filters

The `oxideav-audio-filter` crate provides:

- **Volume** — gain adjustment with configurable scale factor
- **NoiseGate** — threshold-based gate with attack/hold/release
- **Echo** — delay line with feedback
- **Resample** — polyphase windowed-sinc sample rate conversion
- **Spectrogram** — STFT → image (Viridis/Magma colormaps, RGB + PNG output)

### Pixel formats + conversion

The `oxideav-pixfmt` crate is the shared conversion layer for video
codecs. The `PixelFormat` enum covers ~30 first-tier formats (ffmpeg
equivalent names in parentheses):

- RGB family: `Rgb24`, `Bgr24`, `Rgba`, `Bgra`, `Argb`, `Abgr`, plus
  16-bit-per-channel `Rgb48Le` / `Rgba64Le`.
- YUV planar: `Yuv420P` / `Yuv422P` / `Yuv444P` at 8 / 10 / 12-bit,
  plus JPEG-full-range variants (`YuvJ420P`, `YuvJ422P`, `YuvJ444P`).
- YUV semi-planar: `Nv12`, `Nv21`. YUV packed: `Yuyv422`, `Uyvy422`.
- Grayscale: `Gray8`, `Gray10Le`, `Gray12Le`, `Gray16Le`.
- Alpha-bearing: `Ya8`, `Yuva420P`.
- Palette: `Pal8`. 1-bit: `MonoBlack`, `MonoWhite`.

`oxideav_pixfmt::convert(src, dst_format, &ConvertOptions)` handles
the live conversion matrix (RGB all-to-all swizzles, YUV↔RGB under
BT.601 / BT.709 × limited / full range, NV12/NV21 ↔ Yuv420P, Gray ↔
RGB, Rgb48 ↔ Rgb24, Pal8 ↔ RGB with optional dither). Palette
generation via `generate_palette()` offers MedianCut and Uniform
strategies. Dither options: None, 8×8 ordered Bayer, Floyd-Steinberg.

Codecs declare `accepted_pixel_formats` on their `CodecCapabilities`;
the job graph (below) auto-inserts conversion when the upstream
format doesn't match.

### JSON job graph

The `oxideav-job` crate is a declarative way to describe multi-output
transcode pipelines. A job is a JSON object: keys are output
filenames (or reserved sinks like `@null` / `@display`), values
describe tracks grouped by `audio` / `video` / `subtitle` / `all`,
and each track carries a recursive input tree of source refs and
filter / convert nodes.

```json
{
  "threads": 8,
  "@in":       {"all": [{"from": "movie.mp4"}]},
  "out.mkv":   {
    "video": [{"from": "@in", "codec": "h264", "codec_params": {"crf": 23}}],
    "audio": [{"from": "@in", "codec": "flac"}]
  },
  "out.png":   {"video": [{"from": "@in", "convert": "rgba"}]}
}
```

The executor has two modes: **serial** (`threads == 1`) runs one
packet at a time; **pipelined** (`threads ≥ 2`, default when
`available_parallelism()` ≥ 2) spawns one worker thread per stage
per track connected by bounded mpsc channels. The mux/sink loop runs
on the caller's thread so `JobSink` implementations don't need to be
`Send` (the SDL2 player sink in oxideplay stays a single-threaded
object). Both modes produce byte-identical output for deterministic
jobs.

`Decoder` / `Encoder` trait hook: `set_execution_context(&ExecutionContext)`
(default no-op) lets codecs opt into slice- / GOP-parallel work later
without trait churn.

Explicit pixel-format conversion nodes (`{"convert": "yuv420p",
"input": ...}`) fit anywhere in the input tree; the resolver also
auto-inserts a `PixConvert` stage between Decode and Encode when a
codec's `accepted_pixel_formats` list excludes the upstream format.

## Input sources

The source layer decouples I/O from container parsing. Container
demuxers receive an already-opened `Box<dyn ReadSeek>` and never touch
the filesystem directly. The `SourceRegistry` resolves URIs to readers:

| Scheme | Driver | Notes |
|--------|--------|-------|
| bare path / `file://` | built-in | `std::fs::File` |
| `http://` / `https://` | `oxideav-http` (opt-in) | `ureq` + `rustls`, Range-request seeking |

The HTTP driver is off by default in the library (`http` cargo feature)
and on by default in `oxideplay` and `oxideav-cli`.

`BufferedSource` wraps any `ReadSeek` with a prefetch ring buffer
(64 MiB default in oxideplay, configurable via `--buffer-mib`). A
worker thread fills the ring ahead of the read cursor; seeks inside the
window are free.

```
$ oxideav probe https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
Input: https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
Format: mp4
Duration: 00:09:56.46
  Stream #0 [Video]  codec=h264  video 320x180
  Stream #1 [Audio]  codec=aac  audio 2ch @ 48000 Hz
```

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
cargo run -p oxideplay -- https://example.com/video.mp4
```

Keybinds: `q` quit, `space` pause, `← / →` seek ±10 s, `↑ / ↓` seek
±1 min (up = forward, down = back), `pgup / pgdn` seek ±10 min, `*`
volume up, `/` volume down. Works from the SDL window (when a video
stream is present) or from the TTY.

## CLI

`oxideav` command-line verbs: `list`, `probe`, `remux`, `transcode`,
`run`, `validate`, `dry-run`. Inputs can be local paths or HTTP(S)
URLs.

```
$ oxideav list                           # print registered codecs + containers
$ oxideav probe song.flac
$ oxideav transcode song.flac song.wav
$ oxideav remux input.ogg output.mkv
$ oxideav probe https://example.com/video.mp4

# JSON job graph
$ oxideav run job.json
$ oxideav run - < job.json
$ oxideav run --inline '{"out.mkv":{"audio":[{"from":"in.mp3"}]}}'
$ oxideav run --threads 4 job.json        # override thread budget
$ oxideav validate job.json               # check without running
$ oxideav dry-run job.json                # print the resolved DAG
```

`oxideplay --job <file>` runs a job where `@display` / `@out` binds
to the SDL2 player sink; other outputs (file paths) write to disk in
the same run.

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
