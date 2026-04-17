//! Range encoder (RFC 6716 §4.1, inverse of `range_decoder`).
//!
//! Port of libopus `entenc.c`. Mirrors [`crate::range_decoder::RangeDecoder`]
//! bit-for-bit: a roundtrip via a matched encoder/decoder pair reconstructs
//! the exact symbol stream, which is what the CELT encoder needs so the
//! existing decoder can consume our packets.
//!
//! Same two-sided buffer layout as the decoder: main arithmetic-coded
//! symbols grow from the front, raw bits from the back. `done()` flushes
//! remaining state bits and produces a byte-exact output buffer.

use oxideav_core::{Error, Result};

// Match the constants from range_decoder.rs.
const EC_SYM_BITS: u32 = 8;
const EC_CODE_BITS: u32 = 32;
const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;
const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
const EC_UINT_BITS: u32 = 8;
const EC_WINDOW_SIZE: u32 = 32;

/// log2 fractional precision: mirror of the decoder constant.
pub const BITRES: i32 = 3;

/// Range encoder state, matching libopus `ec_ctx`/`ec_enc`.
pub struct RangeEncoder {
    buf: Vec<u8>,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    /// Total whole bits committed (does not include partial bits in `rng`).
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    /// Last-emitted byte that's still subject to carry propagation (-1 = none).
    rem: i32,
    /// Number of buffered 0xff carry-propagating bytes.
    ext: u32,
    error: bool,
}

impl RangeEncoder {
    /// Create a new encoder writing into a buffer of `storage` bytes. The
    /// full output stream (front arithmetic + back raw bits) must fit.
    pub fn new(storage: u32) -> Self {
        Self {
            buf: vec![0u8; storage as usize],
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            // Matches libopus ec_enc_init: -((CODE_BITS-2)%SYM_BITS+1) + CODE_BITS + 1
            nbits_total: (EC_CODE_BITS as i32) + 1
                - ((EC_CODE_BITS as i32 - EC_CODE_EXTRA as i32) / EC_SYM_BITS as i32)
                    * EC_SYM_BITS as i32,
            offs: 0,
            rng: EC_CODE_TOP,
            val: 0,
            rem: -1,
            ext: 0,
            error: false,
        }
    }

    fn write_byte(&mut self, b: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = true;
            return;
        }
        self.buf[self.offs as usize] = (b & 0xff) as u8;
        self.offs += 1;
    }

    fn write_byte_at_end(&mut self, b: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = true;
            return;
        }
        self.end_offs += 1;
        let idx = self.storage - self.end_offs;
        self.buf[idx as usize] = (b & 0xff) as u8;
    }

    /// Carry-propagating output. Port of libopus `ec_enc_carry_out`.
    fn carry_out(&mut self, c: i32) {
        if c as u32 != EC_SYM_MAX {
            // Emit rem + carry bit.
            let carry = (c as u32) >> EC_SYM_BITS;
            if self.rem >= 0 {
                let r = (self.rem as u32).wrapping_add(carry) & EC_SYM_MAX;
                self.write_byte(r);
            }
            if self.ext > 0 {
                let propagated = (EC_SYM_MAX + carry) & EC_SYM_MAX;
                while self.ext > 0 {
                    self.write_byte(propagated);
                    self.ext -= 1;
                }
            }
            self.rem = (c as u32 & EC_SYM_MAX) as i32;
        } else {
            self.ext += 1;
        }
    }

    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            let byte = (self.val >> EC_CODE_SHIFT) as i32;
            self.carry_out(byte);
            self.val = self.val.wrapping_shl(EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS as i32;
        }
    }

    /// Encode one symbol whose CDF places it in `[fl, fh)` out of `ft`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.normalize();
    }

    /// Encode one symbol in a power-of-two CDF (`ft = 1 << bits`).
    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fh));
        }
        self.normalize();
    }

    /// Encode a binary symbol with logp-weighted "1" probability. Matches
    /// `decode_bit_logp` in reverse.
    pub fn encode_bit_logp(&mut self, val: bool, logp: u32) {
        let r = self.rng;
        let s = r >> logp;
        if val {
            // "1" branch: decoder sees d < s.
            self.val = self.val.wrapping_add(r - s);
            self.rng = s;
        } else {
            self.rng = r - s;
        }
        self.normalize();
    }

    /// Encode a symbol via inverse-CDF. `s` is the symbol index in
    /// `icdf[..]` (same semantics as `decode_icdf`).
    pub fn encode_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        let fh = if s > 0 {
            icdf[s - 1] as u32
        } else {
            1u32 << ftb
        };
        let fl = icdf[s] as u32;
        // Match the decoder walk: at step k, `s := r * icdf[k]`; it halts
        // when d >= s, so the winning symbol's range is
        //   val in [r * icdf[s] .. r * icdf[s-1])  (icdf counts down).
        // In encoder terms (fl < fh in the usual sense), the "inner" edge
        // toward 0 is `fh = icdf[s-1]` and the "outer" is `fl = icdf[s]`.
        if s > 0 {
            // val += rng - r * fh; rng = r * (fh - fl)
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(fh)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(fl));
        }
        self.normalize();
    }

    /// Encode a uniform integer in `[0, ft)` (mirrors `decode_uint`).
    pub fn encode_uint(&mut self, v: u32, ft: u32) {
        debug_assert!(ft > 1);
        let ft_minus_1 = ft - 1;
        let ftb = 32 - ft_minus_1.leading_zeros();
        if ftb > EC_UINT_BITS {
            let ftb_extra = ftb - EC_UINT_BITS;
            let ft_top = (ft_minus_1 >> ftb_extra) + 1;
            let high = v >> ftb_extra;
            self.encode(high, high + 1, ft_top);
            self.encode_bits(v & ((1u32 << ftb_extra) - 1), ftb_extra);
        } else {
            self.encode(v, v + 1, ft);
        }
    }

    /// Write `bits` raw bits to the back of the buffer.
    pub fn encode_bits(&mut self, v: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        debug_assert!(bits <= EC_WINDOW_SIZE);
        window |= (v & ((1u64 << bits) - 1) as u32) << used;
        used += bits as i32;
        while used >= EC_SYM_BITS as i32 {
            self.write_byte_at_end(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as i32;
        }
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// Current whole-bit count consumed (mirrors decoder `tell`).
    pub fn tell(&self) -> i32 {
        self.nbits_total - (32 - self.rng.leading_zeros()) as i32
    }

    /// Current 1/8-bit count consumed (mirrors decoder `tell_frac`).
    pub fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = (self.nbits_total as u32) << BITRES;
        let l = 32 - self.rng.leading_zeros();
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        if r > CORRECTION[b as usize] {
            b += 1;
        }
        let l = (l << 3) + b;
        nbits - l
    }

    /// Finalise the stream and return the byte-exact output buffer.
    /// Port of libopus `ec_enc_done`.
    pub fn done(mut self) -> Result<Vec<u8>> {
        // Find the smallest nonzero number of output bits that uniquely
        // identifies a point inside the final range [val, val + rng).
        let mut l = (EC_CODE_BITS - EC_SYM_BITS - 1) as i32 - self.rng.leading_zeros() as i32;
        if l < 0 {
            l = 0;
        }
        // Shift one step further (libopus uses CODE_BITS - SYM_BITS - leading_zeros()).
        // Build mask that covers the uncoded low bits.
        let mut msk = (EC_CODE_TOP - 1) >> l as u32;
        let mut end = (self.val + msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = (self.val + msk) & !msk;
        }

        // Emit `end` high-to-low, byte at a time, through carry_out.
        while l + EC_SYM_BITS as i32 > 0 {
            let byte = (end >> EC_CODE_SHIFT) as i32;
            self.carry_out(byte);
            end = end.wrapping_shl(EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as i32;
        }
        // Resolve any remaining carry buffer.
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        // Flush the last partial back-buffer word, then merge the bit-window
        // into its aligned byte. libopus writes `end_window` bits into the
        // last front-buffer byte; here we mirror that by writing from the
        // end (same physical location).
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used > 0 {
            self.write_byte_at_end(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as i32;
        }

        // Zero out the gap between front and back writes (decoder expects
        // zero-padded interior).
        let written_front = self.offs as usize;
        let written_back = self.end_offs as usize;
        if written_front + written_back < self.storage as usize {
            for b in self
                .buf
                .iter_mut()
                .skip(written_front)
                .take(self.storage as usize - written_front - written_back)
            {
                *b = 0;
            }
        }
        if self.error {
            return Err(Error::invalid(
                "CELT range encoder: ran out of output storage",
            ));
        }
        Ok(self.buf)
    }

    pub fn storage(&self) -> u32 {
        self.storage
    }

    pub fn error(&self) -> bool {
        self.error
    }

    pub fn total_bits(&self) -> u32 {
        self.storage * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;

    #[test]
    fn roundtrip_single_logp_bit() {
        for logp in 1..8u32 {
            for val in [false, true] {
                let mut enc = RangeEncoder::new(8);
                enc.encode_bit_logp(val, logp);
                let buf = enc.done().unwrap();
                let mut dec = RangeDecoder::new(&buf);
                let got = dec.decode_bit_logp(logp);
                assert_eq!(got, val, "mismatch logp={logp} val={val}");
            }
        }
    }

    #[test]
    fn roundtrip_raw_bits() {
        let mut enc = RangeEncoder::new(8);
        enc.encode_bits(0b1011_0101, 8);
        let buf = enc.done().unwrap();
        let mut dec = RangeDecoder::new(&buf);
        let v = dec.decode_bits(8);
        assert_eq!(v, 0b1011_0101);
    }

    #[test]
    fn roundtrip_icdf_small() {
        // A 3-symbol icdf.
        let icdf = [5u8, 2, 0]; // ft = 8
        for s in 0..3 {
            let mut enc = RangeEncoder::new(8);
            enc.encode_icdf(s, &icdf, 3);
            let buf = enc.done().unwrap();
            let mut dec = RangeDecoder::new(&buf);
            let got = dec.decode_icdf(&icdf, 3);
            assert_eq!(got, s);
        }
    }

    #[test]
    fn roundtrip_uint_small() {
        for v in [0u32, 1, 2, 3] {
            let mut enc = RangeEncoder::new(8);
            enc.encode_uint(v, 4);
            let buf = enc.done().unwrap();
            let mut dec = RangeDecoder::new(&buf);
            let got = dec.decode_uint(4);
            assert_eq!(got, v, "mismatch v={v}");
        }
    }

    #[test]
    fn roundtrip_uint_large() {
        for v in [0u32, 500, 1234, 65535] {
            let mut enc = RangeEncoder::new(16);
            enc.encode_uint(v, 65536);
            let buf = enc.done().unwrap();
            let mut dec = RangeDecoder::new(&buf);
            let got = dec.decode_uint(65536);
            assert_eq!(got, v, "mismatch v={v}");
        }
    }

    #[test]
    fn roundtrip_mixed_sequence() {
        let mut enc = RangeEncoder::new(32);
        enc.encode_bit_logp(true, 3);
        enc.encode_bit_logp(false, 1);
        enc.encode_uint(42, 100);
        enc.encode_bits(0xab, 8);
        let buf = enc.done().unwrap();
        let mut dec = RangeDecoder::new(&buf);
        assert!(dec.decode_bit_logp(3));
        assert!(!dec.decode_bit_logp(1));
        assert_eq!(dec.decode_uint(100), 42);
        assert_eq!(dec.decode_bits(8), 0xab);
    }
}
