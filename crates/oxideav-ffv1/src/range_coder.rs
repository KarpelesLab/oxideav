//! FFV1 range coder (RFC 9043 §4.1).
//!
//! Byte-oriented arithmetic coder shared by the encoder and decoder. The
//! fundamental operation is `get_rac`/`put_rac` — a binary symbol whose
//! probability lives in a `state` byte. A "state transition table" maps the
//! old state to a new one after each symbol, approximating adaptive
//! probability estimation.
//!
//! All values quoted in this module come from RFC 9043 Figure 24 (default
//! state transition table). There is no fractional-bit accounting: after
//! writing all symbols the encoder flushes its remaining low register (two
//! bytes is enough).

/// FFV1's default state transition table (Figure 24 of RFC 9043).
pub const DEFAULT_STATE_TRANSITION: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 190, 191, 192, 194,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 202, 204, 205, 206, 207, 208, 209, 209, 210, 211,
    212, 213, 215, 215, 216, 217, 218, 219, 220, 220, 222, 223, 224, 225, 226, 227, 227, 229, 229,
    230, 231, 232, 234, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,
    248, 0, 0, 0, 0, 0, 0, 0,
];

/// Resolved (one_state, zero_state) table pair. `one_state[i]` is the state
/// used after we just coded a `1` given state `i`; `zero_state[i]` is the
/// counterpart for `0`. Derived from a transition table per RFC 9043.
#[derive(Clone)]
pub struct StateTransition {
    pub one_state: [u8; 256],
    pub zero_state: [u8; 256],
}

impl StateTransition {
    pub fn from_table(tbl: &[u8; 256]) -> Self {
        let mut one_state = [0u8; 256];
        let mut zero_state = [0u8; 256];
        one_state[1..256].copy_from_slice(&tbl[1..256]);
        for i in 1..256 {
            // zero_state[i] = 256 - one_state[256-i]
            let mirror = 256usize - i;
            let one = one_state[mirror] as usize;
            zero_state[i] = (256 - one) as u8;
        }
        Self {
            one_state,
            zero_state,
        }
    }

    pub fn default_ffv1() -> Self {
        Self::from_table(&DEFAULT_STATE_TRANSITION)
    }
}

// -----------------------------------------------------------------------
// Decoder
// -----------------------------------------------------------------------

/// Byte-wise range decoder. `pos` walks the input buffer; when it reaches
/// the end, further refill() calls will stall at the current value which is
/// fine — by then the caller should have read all meaningful bits.
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    pos: usize,
    low: u32,
    range: u32,
    transition: StateTransition,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self::with_transition(buf, StateTransition::default_ffv1())
    }

    pub fn with_transition(buf: &'a [u8], transition: StateTransition) -> Self {
        let mut pos = 0usize;
        let b0 = if pos < buf.len() {
            let b = buf[pos];
            pos += 1;
            b
        } else {
            0
        };
        let b1 = if pos < buf.len() {
            let b = buf[pos];
            pos += 1;
            b
        } else {
            0
        };
        let mut low = ((b0 as u32) << 8) | (b1 as u32);
        let range: u32 = 0xFF00;
        if low >= range {
            low = range;
        }
        Self {
            buf,
            pos,
            low,
            range,
            transition,
        }
    }

    /// Current byte offset into the input buffer (useful for slicing the
    /// tail after the range-coded prefix).
    pub fn position(&self) -> usize {
        self.pos
    }

    fn refill(&mut self) {
        if self.range < 0x100 {
            self.range <<= 8;
            self.low <<= 8;
            let b = if self.pos < self.buf.len() {
                let v = self.buf[self.pos];
                self.pos += 1;
                v
            } else {
                0
            };
            self.low |= b as u32;
        }
    }

    /// Decode a single binary symbol using the given state, updating both the
    /// coder and the state byte.
    pub fn get_rac(&mut self, state: &mut u8) -> bool {
        let rangeoff = (self.range * (*state as u32)) >> 8;
        self.range -= rangeoff;
        if self.low < self.range {
            *state = self.transition.zero_state[*state as usize];
            self.refill();
            false
        } else {
            self.low -= self.range;
            self.range = rangeoff;
            *state = self.transition.one_state[*state as usize];
            self.refill();
            true
        }
    }

    /// Decode a symbol using the sign/exponent/mantissa scheme from
    /// §3.8.1.2. `state` is 32 bytes long. Returns the integer value.
    pub fn get_symbol(&mut self, state: &mut [u8; 32], is_signed: bool) -> i32 {
        if self.get_rac(&mut state[0]) {
            return 0;
        }
        let mut e: u32 = 0;
        while self.get_rac(&mut state[1 + e.min(9) as usize]) {
            e += 1;
            if e > 31 {
                // Safety valve — values this large cannot be represented
                // losslessly in i32, so cap and break out.
                break;
            }
        }
        let mut a: i32 = 1;
        for i in (0..e).rev() {
            a = (a << 1) | (self.get_rac(&mut state[22 + i.min(9) as usize]) as i32);
        }
        if is_signed && self.get_rac(&mut state[11 + e.min(10) as usize]) {
            -a
        } else {
            a
        }
    }

    /// Unsigned convenience shortcut — `get_symbol` with `is_signed == false`.
    pub fn get_symbol_u(&mut self, state: &mut [u8; 32]) -> u32 {
        self.get_symbol(state, false) as u32
    }
}

// -----------------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------------

/// Byte-wise range encoder. Appends compressed bytes to `out`.
pub struct RangeEncoder {
    pub out: Vec<u8>,
    low: u32,
    range: u32,
    /// Counts how many 0xFF bytes have been deferred because of potential
    /// carry propagation. Each one will resolve to either 0xFF or 0x00
    /// depending on whether the eventual byte overflows.
    outstanding: u32,
    /// The byte waiting to be shifted out. -1 at start (no buffered byte).
    buffered: i32,
    transition: StateTransition,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self::with_transition(StateTransition::default_ffv1())
    }

    pub fn with_transition(transition: StateTransition) -> Self {
        Self {
            out: Vec::with_capacity(256),
            low: 0,
            range: 0xFF00,
            outstanding: 0,
            buffered: -1,
            transition,
        }
    }

    /// Combined renormalise-and-emit-byte step — FFmpeg's `renorm_encoder`.
    /// Exactly one byte is shifted out of `low` (either immediately committed
    /// via `buffered`, or deferred to `outstanding` if we're in the ambiguous
    /// 0xFF01..=0xFFFF zone), and `range <<= 8` is applied.
    fn shift_low(&mut self) {
        let key = self.low.wrapping_sub(0xFF01);
        if key >= 0xFF {
            let carry = if self.low >= 0x1_0000 { 1u8 } else { 0 };
            if self.buffered >= 0 {
                self.out.push((self.buffered as u8).wrapping_add(carry));
            }
            let fill: u8 = if carry == 0 { 0xFF } else { 0x00 };
            for _ in 0..self.outstanding {
                self.out.push(fill);
            }
            self.outstanding = 0;
            self.buffered = ((self.low >> 8) & 0xFF) as i32;
        } else {
            self.outstanding += 1;
        }
        self.low = (self.low & 0xFF) << 8;
        self.range <<= 8;
    }

    /// Renormalise only when `range` has collapsed below 0x100 — same guard
    /// FFmpeg uses inside `put_rac`.
    fn maybe_renormalize(&mut self) {
        if self.range < 0x100 {
            self.shift_low();
        }
    }

    /// Encode a single binary symbol with the probability encoded in `state`.
    pub fn put_rac(&mut self, state: &mut u8, bit: bool) {
        let rangeoff = (self.range * (*state as u32)) >> 8;
        if bit {
            self.low += self.range - rangeoff;
            self.range = rangeoff;
            *state = self.transition.one_state[*state as usize];
        } else {
            self.range -= rangeoff;
            *state = self.transition.zero_state[*state as usize];
        }
        self.maybe_renormalize();
    }

    /// Encode an integer with the sign/exponent/mantissa scheme.
    pub fn put_symbol(&mut self, state: &mut [u8; 32], value: i32, is_signed: bool) {
        if value == 0 {
            self.put_rac(&mut state[0], true);
            return;
        }
        self.put_rac(&mut state[0], false);
        let a = value.unsigned_abs();
        let e = 31 - a.leading_zeros();
        for i in 0..e {
            self.put_rac(&mut state[1 + i.min(9) as usize], true);
        }
        self.put_rac(&mut state[1 + e.min(9) as usize], false);
        for i in (0..e).rev() {
            let bit = ((a >> i) & 1) != 0;
            self.put_rac(&mut state[22 + i.min(9) as usize], bit);
        }
        if is_signed {
            self.put_rac(&mut state[11 + e.min(10) as usize], value < 0);
        }
    }

    pub fn put_symbol_u(&mut self, state: &mut [u8; 32], value: u32) {
        self.put_symbol(state, value as i32, false);
    }

    /// Flush the internal state. Must be called once at the end.
    ///
    /// Legacy termination path that drops trailing zero bytes — kept for
    /// backward compatibility with internal range-coder unit tests that
    /// expect minimal output. New code should prefer `finish_for_slice`
    /// (FFV1 v1+ slice terminator) or `finish_for_extradata` (FFV1 config
    /// record terminator).
    pub fn finish(mut self) -> Vec<u8> {
        // Five rounds of `shift_low` flush the ambiguous zone thoroughly.
        for _ in 0..5 {
            self.shift_low();
        }
        while self.out.last().copied() == Some(0) {
            self.out.pop();
        }
        self.out
    }

    /// Flush the coder exactly the way FFmpeg's `ff_rac_terminate(c, 0)`
    /// does, for the configuration record. Matches the byte-length that a
    /// conforming FFV1 parser expects.
    pub fn finish_for_extradata(mut self) -> Vec<u8> {
        self.terminate_common();
        self.out
    }

    /// Flush the coder exactly the way FFmpeg's `ff_rac_terminate(c, 1)`
    /// does, for a slice's range-coded data. Emits an extra
    /// `put_rac(state=129, 0)` marker before the common termination.
    pub fn finish_for_slice(mut self) -> Vec<u8> {
        // The `129` marker: `put_rac(c, { 129 }, 0)`. FFmpeg's decoder
        // reads this back via `get_rac({129})` after the last pixel.
        let mut marker_state = 129u8;
        self.put_rac(&mut marker_state, false);
        self.terminate_common();
        self.out
    }

    /// Common tail of both `ff_rac_terminate` variants. Matches FFmpeg's
    /// `range = 0xFF; low += 0xFF; renorm_encoder(c); range = 0xFF;
    /// renorm_encoder(c);`. `shift_low` now embeds the `range <<= 8`
    /// step (mirroring FFmpeg's `renorm_encoder`) so we don't apply it
    /// again here.
    fn terminate_common(&mut self) {
        self.range = 0xFF;
        self.low += 0xFF;
        self.shift_low();
        self.range = 0xFF;
        self.shift_low();
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_tables_are_symmetric() {
        let t = StateTransition::default_ffv1();
        // Spot-check: one_state[128] should match raw transition at 128.
        assert_eq!(t.one_state[128], DEFAULT_STATE_TRANSITION[128]);
        // zero_state[i] = (256 - one_state[256-i]) mod 256 — when one_state is
        // 0 (indicating an unreachable state), the subtraction wraps to 0.
        for i in 1..256usize {
            let one = t.one_state[256 - i] as usize;
            let rhs = (256 - one) & 0xFF;
            assert_eq!(t.zero_state[i] as usize, rhs, "mismatch at i={}", i);
        }
    }

    #[test]
    fn binary_roundtrip_fixed_pattern() {
        let bits = vec![
            true, false, true, true, false, false, true, false, true, true, false, true, false,
            true, false, false, false, true, true, true,
        ];
        let mut enc = RangeEncoder::new();
        let mut state = 128u8;
        for &b in &bits {
            enc.put_rac(&mut state, b);
        }
        let data = enc.finish();
        let mut dec = RangeDecoder::new(&data);
        let mut dstate = 128u8;
        for (i, &b) in bits.iter().enumerate() {
            let got = dec.get_rac(&mut dstate);
            assert_eq!(got, b, "mismatch at index {}", i);
        }
    }

    #[test]
    fn binary_roundtrip_random_bits() {
        // Simple LCG so the test is deterministic.
        let mut x: u32 = 0xdead_beef;
        let mut bits = Vec::with_capacity(2000);
        for _ in 0..2000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            bits.push((x & 1) != 0);
        }
        let mut enc = RangeEncoder::new();
        let mut estate = 128u8;
        for &b in &bits {
            enc.put_rac(&mut estate, b);
        }
        let data = enc.finish();
        let mut dec = RangeDecoder::new(&data);
        let mut dstate = 128u8;
        for (i, &b) in bits.iter().enumerate() {
            let got = dec.get_rac(&mut dstate);
            assert_eq!(got, b, "mismatch at bit {}", i);
        }
    }

    #[test]
    fn symbol_roundtrip_unsigned() {
        let values: Vec<u32> = vec![0, 1, 2, 3, 4, 10, 127, 128, 255, 1024, 65535, 1_000_000];
        let mut enc = RangeEncoder::new();
        let mut state = [128u8; 32];
        for &v in &values {
            enc.put_symbol_u(&mut state, v);
        }
        let data = enc.finish();
        let mut dec = RangeDecoder::new(&data);
        let mut dstate = [128u8; 32];
        for &expected in &values {
            let got = dec.get_symbol_u(&mut dstate);
            assert_eq!(got, expected, "mismatch for value {}", expected);
        }
    }

    #[test]
    fn symbol_roundtrip_signed() {
        let values: Vec<i32> = vec![0, 1, -1, 2, -2, 127, -128, 1024, -1024, 65535, -65535];
        let mut enc = RangeEncoder::new();
        let mut state = [128u8; 32];
        for &v in &values {
            enc.put_symbol(&mut state, v, true);
        }
        let data = enc.finish();
        let mut dec = RangeDecoder::new(&data);
        let mut dstate = [128u8; 32];
        for &expected in &values {
            let got = dec.get_symbol(&mut dstate, true);
            assert_eq!(got, expected, "mismatch for value {}", expected);
        }
    }
}
