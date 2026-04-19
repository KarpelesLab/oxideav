//! Runtime SDL2 loader.
//!
//! Resolves the SDL2 shared library and the small subset of C entry
//! points oxideplay actually needs. We deliberately avoid linking
//! against `sdl2-sys` so the binary still builds (and runs in headless
//! / TUI-only mode) on machines that don't have SDL2 installed.
//!
//! The struct also re-declares the handful of constants and POD structs
//! that we touch through the FFI boundary, so the driver code can stay
//! free of `c_*` types.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::os::raw::c_uchar;

use libloading::{Library, Symbol};

// ---------------------------------------------------------------------------
// SDL2 constants and small structs
// ---------------------------------------------------------------------------

// SDL_Init flags (from SDL.h).
pub const SDL_INIT_AUDIO: u32 = 0x0000_0010;
pub const SDL_INIT_VIDEO: u32 = 0x0000_0020;
pub const SDL_INIT_EVENTS: u32 = 0x0000_4000;

// SDL_AudioFormat (from SDL_audio.h). LSB byte order.
pub const AUDIO_F32LSB: u16 = 0x8120;
pub const AUDIO_F32: u16 = AUDIO_F32LSB;
pub const AUDIO_S16LSB: u16 = 0x8010;
pub const AUDIO_S16: u16 = AUDIO_S16LSB;

// Window flags / position (SDL_video.h).
pub const SDL_WINDOWPOS_CENTERED: c_int = 0x2FFF_0000u32 as c_int;
pub const SDL_WINDOW_RESIZABLE: u32 = 0x0000_0020;

// Texture access (SDL_render.h). STREAMING means lockable + frequently updated.
pub const SDL_TEXTUREACCESS_STREAMING: c_int = 1;

// Pixel formats. These are normally generated through nested macros in
// SDL_pixels.h; we precompute the exact integer values here so we don't
// need to drag those macros into Rust.
//
// SDL_DEFINE_PIXELFOURCC('I','Y','U','V')
//   = 'I' | ('Y' << 8) | ('U' << 16) | ('V' << 24) = 0x5655_5949
pub const SDL_PIXELFORMAT_IYUV: u32 = 0x5655_5949;
//
// SDL_DEFINE_PIXELFORMAT(PIXELTYPE_ARRAYU8, ARRAYORDER_RGB, 0, 24, 3)
//   = (1<<28) | (7<<24) | (1<<20) | (0<<16) | (24<<8) | 3 = 0x1710_1803
pub const SDL_PIXELFORMAT_RGB24: u32 = 0x1710_1803;
//
// SDL_PIXELFORMAT_RGBA32 is little-endian-aliased to ABGR8888:
// SDL_DEFINE_PIXELFORMAT(PIXELTYPE_PACKED32, PACKEDORDER_ABGR=7, PACKEDLAYOUT_8888=6, 32, 4)
//   = (1<<28) | (6<<24) | (7<<20) | (6<<16) | (32<<8) | 4 = 0x1676_2004
pub const SDL_PIXELFORMAT_RGBA32: u32 = 0x1676_2004;

// Event types (from SDL_events.h).
pub const SDL_QUIT: u32 = 0x100;
pub const SDL_KEYDOWN: u32 = 0x300;
pub const SDL_KEYUP: u32 = 0x301;

// Key states.
pub const SDL_PRESSED: u8 = 1;
pub const SDL_RELEASED: u8 = 0;

// Key modifiers (SDL_keycode.h).
pub const KMOD_LSHIFT: u16 = 0x0001;
pub const KMOD_RSHIFT: u16 = 0x0002;
pub const KMOD_SHIFT: u16 = KMOD_LSHIFT | KMOD_RSHIFT;

// Keycodes we actually map. SDLK_<printable> = ascii code; navigation
// keys use SDLK_SCANCODE_MASK = 1 << 30.
pub const SDLK_SCANCODE_MASK: u32 = 1 << 30;
pub const SDLK_ESCAPE: i32 = 0x1B;
pub const SDLK_SPACE: i32 = b' ' as i32;
pub const SDLK_q: i32 = b'q' as i32;
pub const SDLK_f: i32 = b'f' as i32;
pub const SDLK_RIGHT: i32 = (79 | SDLK_SCANCODE_MASK) as i32;
pub const SDLK_LEFT: i32 = (80 | SDLK_SCANCODE_MASK) as i32;
pub const SDLK_DOWN: i32 = (81 | SDLK_SCANCODE_MASK) as i32;
pub const SDLK_UP: i32 = (82 | SDLK_SCANCODE_MASK) as i32;

// SDL_AudioDeviceID is a Uint32; 0 is "no device".
pub type SDL_AudioDeviceID = u32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SDL_AudioSpec {
    pub freq: c_int,
    pub format: u16, // SDL_AudioFormat
    pub channels: u8,
    pub silence: u8,
    pub samples: u16,
    pub padding: u16,
    pub size: u32,
    pub callback: Option<unsafe extern "C" fn(*mut c_void, *mut u8, c_int)>,
    pub userdata: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SDL_Rect {
    pub x: c_int,
    pub y: c_int,
    pub w: c_int,
    pub h: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SDL_Keysym {
    pub scancode: c_int, // SDL_Scancode (enum)
    pub sym: i32,        // SDL_Keycode (i32)
    pub r#mod: u16,
    pub unused: u32,
}

/// Variant of SDL_Event we actually deserialise: keyboard. Layout
/// matches SDL_KeyboardEvent exactly (sizeof must be <= sizeof(SDL_Event)).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SDL_KeyboardEvent {
    pub r#type: u32,
    pub timestamp: u32,
    pub windowID: u32,
    pub state: u8,
    pub repeat: u8,
    pub padding2: u8,
    pub padding3: u8,
    pub keysym: SDL_Keysym,
}

/// Opaque SDL_Event union — we always allocate the full 56 bytes (the
/// padding[] guarantee from SDL_events.h on 64-bit pointer platforms; on
/// 128-bit pointer platforms it's 64). 64 covers both cases safely.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct SDL_Event {
    pub r#type: u32,
    pub padding: [u8; 60], // total = 64 bytes (SDL guarantees ≥ 56 on common targets)
}

impl SDL_Event {
    pub const fn zeroed() -> Self {
        SDL_Event {
            r#type: 0,
            padding: [0u8; 60],
        }
    }

    /// Reinterpret the union as a keyboard event. Caller must check
    /// `type == SDL_KEYDOWN || SDL_KEYUP` first.
    pub unsafe fn as_key(&self) -> &SDL_KeyboardEvent {
        // SAFETY: SDL_KeyboardEvent is one variant of the SDL_Event
        // union and starts at offset 0 with `Uint32 type`. The caller
        // guarantees the discriminant matches.
        &*(self as *const SDL_Event as *const SDL_KeyboardEvent)
    }
}

// ---------------------------------------------------------------------------
// FFI signature aliases (one per function we resolve).
// ---------------------------------------------------------------------------

pub type Fn_SDL_Init = unsafe extern "C" fn(flags: u32) -> c_int;
pub type Fn_SDL_Quit = unsafe extern "C" fn();
pub type Fn_SDL_GetError = unsafe extern "C" fn() -> *const c_char;

pub type Fn_SDL_OpenAudioDevice = unsafe extern "C" fn(
    device: *const c_char,
    iscapture: c_int,
    desired: *const SDL_AudioSpec,
    obtained: *mut SDL_AudioSpec,
    allowed_changes: c_int,
) -> SDL_AudioDeviceID;
pub type Fn_SDL_PauseAudioDevice = unsafe extern "C" fn(dev: SDL_AudioDeviceID, pause_on: c_int);
pub type Fn_SDL_CloseAudioDevice = unsafe extern "C" fn(dev: SDL_AudioDeviceID);
pub type Fn_SDL_QueueAudio =
    unsafe extern "C" fn(dev: SDL_AudioDeviceID, data: *const c_void, len: u32) -> c_int;
pub type Fn_SDL_GetQueuedAudioSize = unsafe extern "C" fn(dev: SDL_AudioDeviceID) -> u32;
pub type Fn_SDL_ClearQueuedAudio = unsafe extern "C" fn(dev: SDL_AudioDeviceID);

pub type Fn_SDL_CreateWindow = unsafe extern "C" fn(
    title: *const c_char,
    x: c_int,
    y: c_int,
    w: c_int,
    h: c_int,
    flags: u32,
) -> *mut c_void;
pub type Fn_SDL_DestroyWindow = unsafe extern "C" fn(window: *mut c_void);
pub type Fn_SDL_SetWindowFullscreen =
    unsafe extern "C" fn(window: *mut c_void, flags: u32) -> c_int;
pub type Fn_SDL_GetWindowFlags = unsafe extern "C" fn(window: *mut c_void) -> u32;

/// `SDL_WINDOW_FULLSCREEN_DESKTOP` — borderless fullscreen at the
/// current desktop resolution (no mode switch). Combined form of
/// `SDL_WINDOW_FULLSCREEN | 0x1000`.
pub const SDL_WINDOW_FULLSCREEN_DESKTOP: u32 = 0x0000_1001;
pub type Fn_SDL_CreateRenderer =
    unsafe extern "C" fn(window: *mut c_void, index: c_int, flags: u32) -> *mut c_void;
pub type Fn_SDL_DestroyRenderer = unsafe extern "C" fn(renderer: *mut c_void);
pub type Fn_SDL_CreateTexture = unsafe extern "C" fn(
    renderer: *mut c_void,
    format: u32,
    access: c_int,
    w: c_int,
    h: c_int,
) -> *mut c_void;
pub type Fn_SDL_DestroyTexture = unsafe extern "C" fn(texture: *mut c_void);
pub type Fn_SDL_UpdateYUVTexture = unsafe extern "C" fn(
    texture: *mut c_void,
    rect: *const SDL_Rect,
    y_plane: *const c_uchar,
    y_pitch: c_int,
    u_plane: *const c_uchar,
    u_pitch: c_int,
    v_plane: *const c_uchar,
    v_pitch: c_int,
) -> c_int;
pub type Fn_SDL_UpdateTexture = unsafe extern "C" fn(
    texture: *mut c_void,
    rect: *const SDL_Rect,
    pixels: *const c_void,
    pitch: c_int,
) -> c_int;
pub type Fn_SDL_RenderClear = unsafe extern "C" fn(renderer: *mut c_void) -> c_int;
pub type Fn_SDL_RenderCopy = unsafe extern "C" fn(
    renderer: *mut c_void,
    texture: *mut c_void,
    srcrect: *const SDL_Rect,
    dstrect: *const SDL_Rect,
) -> c_int;
pub type Fn_SDL_RenderPresent = unsafe extern "C" fn(renderer: *mut c_void);
pub type Fn_SDL_GetRendererOutputSize =
    unsafe extern "C" fn(renderer: *mut c_void, w: *mut c_int, h: *mut c_int) -> c_int;

pub type Fn_SDL_PollEvent = unsafe extern "C" fn(event: *mut SDL_Event) -> c_int;
pub type Fn_SDL_PumpEvents = unsafe extern "C" fn();
pub type Fn_SDL_Delay = unsafe extern "C" fn(ms: c_uint);

// ---------------------------------------------------------------------------
// Loaded library + bound symbols
// ---------------------------------------------------------------------------

/// Holds the dynamically-loaded SDL2 library plus typed function
/// pointers. Drop order matters: the function pointers (`Fn_…`) all
/// reference code inside `_lib`, so `_lib` must outlive every call —
/// which it does, because it's owned by this struct alongside them and
/// dropped last.
pub struct Sdl2Lib {
    _lib: Library,

    pub SDL_Init: Fn_SDL_Init,
    pub SDL_Quit: Fn_SDL_Quit,
    pub SDL_GetError: Fn_SDL_GetError,

    pub SDL_OpenAudioDevice: Fn_SDL_OpenAudioDevice,
    pub SDL_PauseAudioDevice: Fn_SDL_PauseAudioDevice,
    pub SDL_CloseAudioDevice: Fn_SDL_CloseAudioDevice,
    pub SDL_QueueAudio: Fn_SDL_QueueAudio,
    pub SDL_GetQueuedAudioSize: Fn_SDL_GetQueuedAudioSize,
    pub SDL_ClearQueuedAudio: Fn_SDL_ClearQueuedAudio,

    pub SDL_CreateWindow: Fn_SDL_CreateWindow,
    pub SDL_DestroyWindow: Fn_SDL_DestroyWindow,
    pub SDL_SetWindowFullscreen: Fn_SDL_SetWindowFullscreen,
    pub SDL_GetWindowFlags: Fn_SDL_GetWindowFlags,
    pub SDL_CreateRenderer: Fn_SDL_CreateRenderer,
    pub SDL_DestroyRenderer: Fn_SDL_DestroyRenderer,
    pub SDL_CreateTexture: Fn_SDL_CreateTexture,
    pub SDL_DestroyTexture: Fn_SDL_DestroyTexture,
    pub SDL_UpdateYUVTexture: Fn_SDL_UpdateYUVTexture,
    pub SDL_UpdateTexture: Fn_SDL_UpdateTexture,
    pub SDL_RenderClear: Fn_SDL_RenderClear,
    pub SDL_RenderCopy: Fn_SDL_RenderCopy,
    pub SDL_RenderPresent: Fn_SDL_RenderPresent,
    pub SDL_GetRendererOutputSize: Fn_SDL_GetRendererOutputSize,

    pub SDL_PollEvent: Fn_SDL_PollEvent,
    pub SDL_PumpEvents: Fn_SDL_PumpEvents,
    pub SDL_Delay: Fn_SDL_Delay,
}

/// Filenames we try in order. First hit wins. The list covers the
/// "official" soname on Linux distros, the unversioned symlink, the
/// FreeBSD-style name, the macOS dylib, and Windows.
const CANDIDATE_NAMES: &[&str] = &[
    "libSDL2-2.0.so.0",
    "libSDL2-2.0.so",
    "libSDL2.so",
    "libSDL2.dylib",
    "SDL2.dll",
];

impl Sdl2Lib {
    /// Walk through known SDL2 library names and return the first one
    /// that loads + has all the symbols we need. Returns the last error
    /// if every candidate fails.
    pub fn try_load() -> Result<Self, libloading::Error> {
        Self::try_load_from(CANDIDATE_NAMES)
    }

    /// Test hook: try a custom list of library candidates. Used by the
    /// unit tests to verify the "no SDL2 found" failure path without
    /// needing to actually unload SDL2 from the system.
    pub fn try_load_from(candidates: &[&str]) -> Result<Self, libloading::Error> {
        let mut last_err: Option<libloading::Error> = None;
        for name in candidates {
            // SAFETY: dlopen()ing a system shared library is inherently
            // unsafe (constructors run, can mutate process state). We
            // accept this for the rendering / audio backend.
            let lib = unsafe { Library::new(name) };
            match lib {
                Ok(lib) => match unsafe { Self::bind(lib) } {
                    Ok(s) => return Ok(s),
                    Err(e) => last_err = Some(e),
                },
                Err(e) => last_err = Some(e),
            }
        }
        // Synthesise a "library not found" error if the candidate list
        // was somehow empty (it isn't, but be defensive).
        Err(last_err.unwrap_or(libloading::Error::DlOpenUnknown))
    }

    /// Resolve every entry point we use. If any one is missing the
    /// whole load fails — there's no "partial SDL2" that would let us
    /// run sensibly, so it's all-or-nothing.
    unsafe fn bind(lib: Library) -> Result<Self, libloading::Error> {
        // The `Library::get<T>(...)` API returns a `Symbol<'lib, T>`; we
        // then deref + copy out the function pointer. The lifetime of
        // each pointer is tied to `lib`, which we move into the struct
        // alongside them — so they remain valid for the lifetime of
        // `Sdl2Lib`.
        macro_rules! sym {
            ($ty:ty, $name:literal) => {{
                let s: Symbol<$ty> = lib.get(concat!($name, "\0").as_bytes())?;
                *s
            }};
        }

        let s = Self {
            SDL_Init: sym!(Fn_SDL_Init, "SDL_Init"),
            SDL_Quit: sym!(Fn_SDL_Quit, "SDL_Quit"),
            SDL_GetError: sym!(Fn_SDL_GetError, "SDL_GetError"),
            SDL_OpenAudioDevice: sym!(Fn_SDL_OpenAudioDevice, "SDL_OpenAudioDevice"),
            SDL_PauseAudioDevice: sym!(Fn_SDL_PauseAudioDevice, "SDL_PauseAudioDevice"),
            SDL_CloseAudioDevice: sym!(Fn_SDL_CloseAudioDevice, "SDL_CloseAudioDevice"),
            SDL_QueueAudio: sym!(Fn_SDL_QueueAudio, "SDL_QueueAudio"),
            SDL_GetQueuedAudioSize: sym!(Fn_SDL_GetQueuedAudioSize, "SDL_GetQueuedAudioSize"),
            SDL_ClearQueuedAudio: sym!(Fn_SDL_ClearQueuedAudio, "SDL_ClearQueuedAudio"),
            SDL_CreateWindow: sym!(Fn_SDL_CreateWindow, "SDL_CreateWindow"),
            SDL_DestroyWindow: sym!(Fn_SDL_DestroyWindow, "SDL_DestroyWindow"),
            SDL_SetWindowFullscreen: sym!(
                Fn_SDL_SetWindowFullscreen,
                "SDL_SetWindowFullscreen"
            ),
            SDL_GetWindowFlags: sym!(Fn_SDL_GetWindowFlags, "SDL_GetWindowFlags"),
            SDL_CreateRenderer: sym!(Fn_SDL_CreateRenderer, "SDL_CreateRenderer"),
            SDL_DestroyRenderer: sym!(Fn_SDL_DestroyRenderer, "SDL_DestroyRenderer"),
            SDL_CreateTexture: sym!(Fn_SDL_CreateTexture, "SDL_CreateTexture"),
            SDL_DestroyTexture: sym!(Fn_SDL_DestroyTexture, "SDL_DestroyTexture"),
            SDL_UpdateYUVTexture: sym!(Fn_SDL_UpdateYUVTexture, "SDL_UpdateYUVTexture"),
            SDL_UpdateTexture: sym!(Fn_SDL_UpdateTexture, "SDL_UpdateTexture"),
            SDL_RenderClear: sym!(Fn_SDL_RenderClear, "SDL_RenderClear"),
            SDL_RenderCopy: sym!(Fn_SDL_RenderCopy, "SDL_RenderCopy"),
            SDL_RenderPresent: sym!(Fn_SDL_RenderPresent, "SDL_RenderPresent"),
            SDL_GetRendererOutputSize: sym!(
                Fn_SDL_GetRendererOutputSize,
                "SDL_GetRendererOutputSize"
            ),
            SDL_PollEvent: sym!(Fn_SDL_PollEvent, "SDL_PollEvent"),
            SDL_PumpEvents: sym!(Fn_SDL_PumpEvents, "SDL_PumpEvents"),
            SDL_Delay: sym!(Fn_SDL_Delay, "SDL_Delay"),
            _lib: lib,
        };
        Ok(s)
    }

    /// Fetch the last SDL error and convert to a Rust `String`. Returns
    /// the empty string if there's no message.
    pub fn last_error(&self) -> String {
        // SAFETY: SDL_GetError returns a pointer to a NUL-terminated,
        // thread-local C string that's valid until the next SDL call on
        // the same thread. We copy it out immediately.
        unsafe {
            let p = (self.SDL_GetError)();
            if p.is_null() {
                return String::new();
            }
            let cstr = std::ffi::CStr::from_ptr(p);
            cstr.to_string_lossy().into_owned()
        }
    }
}

// SDL2 itself isn't thread-safe in general, but the Library handle is.
// We use the loader from a single thread (the main thread) so the
// auto-traits are fine; we just need to make sure the lib field is the
// last thing dropped. (Dropped last == declared last.)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keycode_constants_match_sdl_definitions() {
        assert_eq!(SDLK_RIGHT, 0x4000_004F);
        assert_eq!(SDLK_LEFT, 0x4000_0050);
        assert_eq!(SDLK_DOWN, 0x4000_0051);
        assert_eq!(SDLK_UP, 0x4000_0052);
        assert_eq!(SDLK_q, 0x71);
        assert_eq!(SDLK_ESCAPE, 0x1B);
        assert_eq!(SDLK_SPACE, 0x20);
    }

    #[test]
    fn pixel_format_constants_match_sdl_definitions() {
        // 'I' | ('Y'<<8) | ('U'<<16) | ('V'<<24)
        assert_eq!(SDL_PIXELFORMAT_IYUV, 0x5655_5949);
        assert_eq!(SDL_PIXELFORMAT_RGB24, 0x1710_1803);
        assert_eq!(SDL_PIXELFORMAT_RGBA32, 0x1676_2004);
    }

    #[test]
    fn sdl_event_is_at_least_56_bytes() {
        assert!(std::mem::size_of::<SDL_Event>() >= 56);
        assert!(std::mem::size_of::<SDL_KeyboardEvent>() <= std::mem::size_of::<SDL_Event>());
    }

    #[test]
    fn try_load_returns_err_when_library_missing() {
        // None of these will exist on any reasonable system.
        let candidates = &[
            "libdefinitely-not-a-real-library-1.so.0",
            "libdefinitely-not-a-real-library-2.so",
        ];
        let res = Sdl2Lib::try_load_from(candidates);
        assert!(res.is_err(), "expected Err, got Ok");
    }
}
