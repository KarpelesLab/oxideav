//! Variable-width LZW encoder/decoder for GIF.
//!
//! GIF uses a classic LZW with a handful of notable quirks:
//!
//! * Minimum code size lives in a header byte. The initial code width
//!   starts at `min_code_size + 1` (to accommodate the clear code and
//!   EOI code, which occupy the two entries immediately past the
//!   alphabet).
//! * Two special codes are pre-reserved:
//!     * `clear = 1 << min_code_size` — reset the dictionary and the
//!       code width.
//!     * `eoi = clear + 1` — end of stream.
//! * Code width grows by 1 after the encoder emits a code whose value
//!   is `(1 << code_width) - 1`, up to a hard cap of 12 bits. Once the
//!   dictionary is full (next available index would be 4096), the
//!   encoder emits a clear code and resets.
//! * Codes are packed LSB-first into the output byte stream — lowest
//!   bit of the code goes to the lowest bit of the current partial byte
//!   first.
//!
//! Decoder implementation is the classic KwKwK loop: when a received
//! code is exactly the next dictionary entry, emit `prefix + prefix[0]`.
//!
//! The compliance corner: some encoders "defer" the code-width increase
//! — when reading we accept the code that exactly matches the boundary
//! (`code == 2^width`) at the width that was valid when it was written.
//! This decoder follows the common convention that matches libgif and
//! ffmpeg: after adding an entry, increase the width *before* reading the
//! next code when `dict_len == (1 << current_width)` and `width < 12`.
//!
//! Both sides restrict code width to `2..=12` bits.

use oxideav_core::{Error, Result};

const MAX_CODE_WIDTH: u8 = 12;
const MAX_DICT_LEN: usize = 1 << MAX_CODE_WIDTH; // 4096

/// One-stop constructor — groups the encoder/decoder factory methods.
pub struct Lzw;

impl Lzw {
    /// Build an LZW encoder for the given minimum code size.
    ///
    /// `min_code_size` must satisfy `2..=11`: GIF requires the initial
    /// code width (`min_code_size + 1`) to leave room for the two
    /// reserved codes (clear + EOI) and to stay ≤ 12 bits.
    pub fn encoder(min_code_size: u8) -> Result<LzwEncoder> {
        LzwEncoder::new(min_code_size)
    }

    /// Build an LZW decoder for the given minimum code size.
    pub fn decoder(min_code_size: u8) -> Result<LzwDecoder> {
        LzwDecoder::new(min_code_size)
    }
}

// ---- Encoder -------------------------------------------------------------

/// Streaming LZW encoder.
pub struct LzwEncoder {
    min_code_size: u8,
    code_width: u8,
    clear_code: u16,
    eoi_code: u16,
    /// (prefix_code, next_byte) -> child code
    ///
    /// We use a flat vec keyed by `prefix * 256 + byte`. Capacity caps
    /// out at `MAX_DICT_LEN * 256` entries; reusing a single allocation
    /// keeps the hot path cache-friendly. A zero entry means "missing"
    /// — real codes are offset by 1 in the table.
    dict: Vec<u16>,
    dict_len: u16,
    current: Option<u16>,
    bw: BitWriter,
}

impl LzwEncoder {
    fn new(min_code_size: u8) -> Result<Self> {
        if !(2..=11).contains(&min_code_size) {
            return Err(Error::invalid(format!(
                "LZW: min_code_size {} out of range 2..=11",
                min_code_size
            )));
        }
        let clear_code = 1u16 << min_code_size;
        let eoi_code = clear_code + 1;
        let code_width = min_code_size + 1;
        let mut enc = Self {
            min_code_size,
            code_width,
            clear_code,
            eoi_code,
            dict: vec![0; MAX_DICT_LEN * 256],
            dict_len: eoi_code + 1,
            current: None,
            bw: BitWriter::new(),
        };
        enc.reset_dict();
        Ok(enc)
    }

    fn reset_dict(&mut self) {
        for slot in self.dict.iter_mut() {
            *slot = 0;
        }
        self.dict_len = self.eoi_code + 1;
        self.code_width = self.min_code_size + 1;
    }

    fn dict_get(&self, prefix: u16, byte: u8) -> Option<u16> {
        let slot = self.dict[prefix as usize * 256 + byte as usize];
        if slot == 0 {
            None
        } else {
            Some(slot - 1)
        }
    }

    fn dict_insert(&mut self, prefix: u16, byte: u8, code: u16) {
        self.dict[prefix as usize * 256 + byte as usize] = code + 1;
    }

    /// Compress a chunk of indices into `out`.
    ///
    /// On the very first call the encoder emits a leading clear code,
    /// then standard LZW. Call [`finish`](Self::finish) at end to flush
    /// the final pending code, emit EOI, and pad to a byte boundary.
    pub fn write(&mut self, indices: &[u8], out: &mut Vec<u8>) {
        if self.current.is_none() && !indices.is_empty() {
            // Emit the initial clear code to match libgif's output shape.
            self.bw.put(self.clear_code as u32, self.code_width, out);
            self.current = Some(indices[0] as u16);
            self.encode_remaining(&indices[1..], out);
        } else {
            self.encode_remaining(indices, out);
        }
    }

    fn encode_remaining(&mut self, indices: &[u8], out: &mut Vec<u8>) {
        for &b in indices {
            let pref = self.current.unwrap_or(b as u16);
            if self.current.is_none() {
                self.current = Some(b as u16);
                continue;
            }
            match self.dict_get(pref, b) {
                Some(c) => {
                    self.current = Some(c);
                }
                None => {
                    // Emit `pref`, extend dictionary.
                    self.bw.put(pref as u32, self.code_width, out);
                    if self.dict_len < MAX_DICT_LEN as u16 {
                        self.dict_insert(pref, b, self.dict_len);
                        self.dict_len += 1;
                        // Grow width after emission if next insertion
                        // will need it. The "boundary" convention is
                        // post-insert: if the insert brought dict_len to
                        // exactly (1 << code_width), bump the width.
                        if self.dict_len == (1u16 << self.code_width)
                            && self.code_width < MAX_CODE_WIDTH
                        {
                            self.code_width += 1;
                        }
                    } else {
                        // Dictionary full — emit clear and reset.
                        self.bw.put(self.clear_code as u32, self.code_width, out);
                        self.reset_dict();
                    }
                    self.current = Some(b as u16);
                }
            }
        }
    }

    /// Emit the final pending code, then EOI, then flush the bit buffer.
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        if let Some(c) = self.current.take() {
            self.bw.put(c as u32, self.code_width, out);
        } else {
            // Degenerate empty input — still emit a clear so the output
            // is a well-formed LZW stream.
            self.bw.put(self.clear_code as u32, self.code_width, out);
        }
        self.bw.put(self.eoi_code as u32, self.code_width, out);
        self.bw.flush(out);
    }
}

// ---- Decoder -------------------------------------------------------------

/// One-shot LZW decoder.
pub struct LzwDecoder {
    min_code_size: u8,
    clear_code: u16,
    eoi_code: u16,
}

impl LzwDecoder {
    fn new(min_code_size: u8) -> Result<Self> {
        if !(2..=11).contains(&min_code_size) {
            return Err(Error::invalid(format!(
                "LZW: min_code_size {} out of range 2..=11",
                min_code_size
            )));
        }
        let clear_code = 1u16 << min_code_size;
        let eoi_code = clear_code + 1;
        Ok(Self {
            min_code_size,
            clear_code,
            eoi_code,
        })
    }

    /// Decompress `compressed` into a freshly allocated index buffer.
    /// Accepts input that terminates at EOI or simply runs out of bytes
    /// (we pad the final code with zeros, matching libgif behaviour).
    pub fn read(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut br = BitReader::new(compressed);
        let mut width = self.min_code_size + 1;
        // Dictionary: each entry stores (prefix_code, first_byte, length).
        // Reconstructing the full byte sequence at decode time is cheap
        // because we walk back through prefixes into a small scratch.
        let mut prefix: Vec<u16> = Vec::with_capacity(MAX_DICT_LEN);
        let mut suffix: Vec<u8> = Vec::with_capacity(MAX_DICT_LEN);
        let mut first_byte: Vec<u8> = Vec::with_capacity(MAX_DICT_LEN);

        // Init dictionary.
        let init_entries = (self.eoi_code + 1) as usize;
        for i in 0..init_entries {
            prefix.push(0xFFFF);
            suffix.push(if (i as u16) < self.clear_code {
                i as u8
            } else {
                0
            });
            first_byte.push(if (i as u16) < self.clear_code {
                i as u8
            } else {
                0
            });
        }

        let mut prev_code: Option<u16> = None;
        let mut scratch: Vec<u8> = Vec::with_capacity(MAX_DICT_LEN);

        loop {
            let code = match br.get(width) {
                Some(c) => c as u16,
                None => break,
            };
            if code == self.clear_code {
                prefix.truncate(init_entries);
                suffix.truncate(init_entries);
                first_byte.truncate(init_entries);
                width = self.min_code_size + 1;
                prev_code = None;
                continue;
            }
            if code == self.eoi_code {
                break;
            }

            let dict_len = prefix.len() as u16;
            let (fb, bytes) = if code < dict_len {
                // Known code — walk its string.
                decode_string(&prefix, &suffix, code, &mut scratch);
                (first_byte[code as usize], &scratch[..])
            } else if code == dict_len {
                // KwKwK: special-case the not-yet-added entry.
                let p = prev_code
                    .ok_or_else(|| Error::invalid("LZW: KwKwK code with no previous code"))?;
                decode_string(&prefix, &suffix, p, &mut scratch);
                let fb = scratch[0];
                scratch.push(fb);
                (fb, &scratch[..])
            } else {
                return Err(Error::invalid(format!(
                    "LZW: code {} past dictionary length {}",
                    code, dict_len
                )));
            };
            out.extend_from_slice(bytes);

            if let Some(p) = prev_code {
                // Add new dictionary entry p + first_byte(code) — but
                // only if we have room.
                if prefix.len() < MAX_DICT_LEN {
                    prefix.push(p);
                    suffix.push(fb);
                    first_byte.push(first_byte[p as usize]);
                    // GIF LZW decoder off-by-one: the decoder's
                    // dict_len lags the encoder's by exactly one entry
                    // (the decoder doesn't insert on the very first
                    // code after a clear, while the encoder inserts
                    // on every emit). To keep the code width in sync,
                    // bump when the next insert WILL hit `1 << width`,
                    // which is equivalent to `dict_len + 1 ==
                    // 1 << width` right now.
                    if prefix.len() + 1 == (1usize << width) && width < MAX_CODE_WIDTH {
                        width += 1;
                    }
                }
            }
            prev_code = Some(code);
        }
        Ok(out)
    }
}

/// Walk a dictionary entry back to its root, writing the byte sequence
/// into `scratch` (left-to-right).
fn decode_string(prefix: &[u16], suffix: &[u8], mut code: u16, scratch: &mut Vec<u8>) {
    scratch.clear();
    // Walk chain from deepest byte back to root.
    loop {
        scratch.push(suffix[code as usize]);
        if prefix[code as usize] == 0xFFFF {
            break;
        }
        code = prefix[code as usize];
    }
    scratch.reverse();
}

// ---- Bit reader / writer (LSB-first) -------------------------------------

struct BitWriter {
    buf: u32,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self { buf: 0, nbits: 0 }
    }

    fn put(&mut self, code: u32, width: u8, out: &mut Vec<u8>) {
        self.buf |= (code & ((1u32 << width) - 1)) << self.nbits;
        self.nbits += width;
        while self.nbits >= 8 {
            out.push((self.buf & 0xFF) as u8);
            self.buf >>= 8;
            self.nbits -= 8;
        }
    }

    fn flush(&mut self, out: &mut Vec<u8>) {
        if self.nbits > 0 {
            out.push((self.buf & 0xFF) as u8);
        }
        self.buf = 0;
        self.nbits = 0;
    }
}

struct BitReader<'a> {
    src: &'a [u8],
    pos: usize,
    buf: u32,
    nbits: u8,
}

impl<'a> BitReader<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            buf: 0,
            nbits: 0,
        }
    }

    fn get(&mut self, width: u8) -> Option<u32> {
        while self.nbits < width {
            if self.pos >= self.src.len() {
                // Pad with zeros to allow reading a final partial code.
                if self.nbits == 0 {
                    return None;
                }
                break;
            }
            self.buf |= (self.src[self.pos] as u32) << self.nbits;
            self.pos += 1;
            self.nbits += 8;
        }
        if self.nbits == 0 {
            return None;
        }
        let take = width.min(self.nbits);
        let mask = (1u32 << take) - 1;
        let v = self.buf & mask;
        self.buf >>= take;
        self.nbits -= take;
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(min_code_size: u8, input: &[u8]) {
        let mut enc = Lzw::encoder(min_code_size).unwrap();
        let mut out = Vec::new();
        enc.write(input, &mut out);
        enc.finish(&mut out);
        let dec = Lzw::decoder(min_code_size).unwrap();
        let decoded = dec.read(&out).unwrap();
        assert_eq!(decoded.as_slice(), input);
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(8, &[]);
    }

    #[test]
    fn roundtrip_single_byte() {
        roundtrip(8, &[0x42]);
    }

    #[test]
    fn roundtrip_short() {
        roundtrip(8, b"hello gif");
    }

    #[test]
    fn roundtrip_monotonous() {
        let buf: Vec<u8> = std::iter::repeat(7u8).take(4000).collect();
        roundtrip(3, &buf);
    }

    #[test]
    fn roundtrip_triggers_dict_reset() {
        // Pseudo-random sequence long enough to fill the dictionary.
        let mut buf = Vec::with_capacity(20_000);
        let mut state: u32 = 0xDEADBEEF;
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            buf.push(((state >> 16) & 0xFF) as u8);
        }
        roundtrip(8, &buf);
    }
}
