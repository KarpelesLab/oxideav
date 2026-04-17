# oxideplay

Reference media player for the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework. Uses the library's pure-Rust demuxers + decoders and hands decoded
frames to SDL2 for audio output and (optionally) video display.

## Build requirement: SDL2

Unlike the rest of the workspace, this binary crate links against SDL2 via
`rust-sdl2` + `sdl2-sys` (C). Install SDL2 first:

- **Gentoo**: `sudo emerge media-libs/libsdl2`
- **Debian / Ubuntu**: `sudo apt install libsdl2-dev`
- **Fedora**: `sudo dnf install SDL2-devel`
- **macOS (Homebrew)**: `brew install sdl2`

## Usage

```sh
cargo run -p oxideplay -- path/to/file.flac
cargo run -p oxideplay -- --dry-run path/to/file.mp4   # probe & exit
cargo run -p oxideplay -- --no-video path/to/file.mp4  # audio only
```

Keybinds (same in window and TUI):

| Key | Action |
| --- | --- |
| `q`, `Esc` | Quit |
| `space` | Pause / resume |
| `←` / `→` | Seek ±5 s |
| `shift+←` / `shift+→` | Seek ±30 s |
| `↑` / `↓` | Volume ±5 % |

When stdout is a TTY, a one-line status bar is shown. When it's piped,
a simple progress message is emitted to stderr every ~1 s instead.
