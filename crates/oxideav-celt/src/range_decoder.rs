//! Range decoder (RFC 6716 §4.1).
//!
//! The entire Opus bitstream — both SILK and CELT paths — is wrapped in a
//! single arithmetic coder: a binary range coder with a 32-bit internal
//! state. This module is a faithful port of the libopus reference
//! implementation (entdec.c) — the spec text is the same algorithm, but
//! libopus is the bit-exact gold reference.
//!
//! The coder has two independent "sides" that share the same buffer but grow
//! toward each other:
//!
//! 1. The main front-loaded **range coder** stream (read with
//!    [`RangeDecoder::decode_icdf`], [`RangeDecoder::decode_uint`], etc.).
//! 2. A back-loaded **raw bits** stream (read with
//!    [`RangeDecoder::decode_bits`]). CELT stores values that don't benefit
//!    from entropy coding here — e.g. fine-energy LSBs and PVQ raw bits.
//!
//! Both sides converge on the same total bit budget. The `tell` methods
//! report the *total* bits consumed by both sides, which is what the CELT
//! allocator uses to decide whether it still has budget for another symbol.

use oxideav_core::{Error, Result};

// ---- Constants from libopus mfrngcod.h --------------------------------------

const EC_SYM_BITS: u32 = 8;
const EC_CODE_BITS: u32 = 32;
const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;
const EC_WINDOW_SIZE: i32 = 32;
const EC_UINT_BITS: u32 = 8;

/// log2 fractional precision: results of ec_tell_frac() are in 1/8 bit units.
pub const BITRES: i32 = 3;

/// Range decoder state — exact 1:1 mirror of libopus `ec_ctx`/`ec_dec`.
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    /// Total whole bits consumed (does not include partial bits in `rng`).
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    /// Saved normalization factor from `ec_decode()` — used by `ec_dec_update`.
    ext: u32,
    /// Buffered symbol awaiting renormalization.
    rem: i32,
    error: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Initialize a range decoder over `buf` (matches libopus `ec_dec_init`).
    pub fn new(buf: &'a [u8]) -> Self {
        let storage = buf.len() as u32;
        let mut d = Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (EC_CODE_BITS as i32) + 1
                - ((EC_CODE_BITS as i32 - EC_CODE_EXTRA as i32) / EC_SYM_BITS as i32)
                    * EC_SYM_BITS as i32,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: false,
        };
        d.rem = d.read_byte() as i32;
        // val = rng - 1 - (rem >> (SYM_BITS - CODE_EXTRA))
        d.val = d.rng - 1 - ((d.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        d.normalize();
        d
    }

    fn read_byte(&mut self) -> u8 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize];
            self.offs += 1;
            b
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u8 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize]
        } else {
            0
        }
    }

    /// Renormalize so that `rng` lies in the high-order symbol (libopus
    /// `ec_dec_normalize`).
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS as i32;
            self.rng <<= EC_SYM_BITS;
            // Use up the remaining bits from our last symbol.
            let sym = self.rem;
            // Read the next value from the input.
            self.rem = self.read_byte() as i32;
            // Take the rest of the bits we need from this new symbol.
            let combined = (sym << EC_SYM_BITS) | self.rem;
            let sym_extra =
                ((combined as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA)) & ((1u32 << 8) - 1);
            // val = ((val << SYM_BITS) + (SYM_MAX & ~sym_extra)) & (CODE_TOP - 1)
            self.val = (self.val.wrapping_shl(EC_SYM_BITS)).wrapping_add(EC_SYM_MAX & !sym_extra)
                & (EC_CODE_TOP - 1);
        }
    }

    /// `ec_decode` (RFC §4.1.3): return the "fractional" value used to look
    /// up a symbol in a CDF whose total is `ft`.
    pub fn decode(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 0);
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// `ec_decode_bin` — same as `decode` but ft is `1 << bits`.
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1u32 << bits) - (s + 1).min(1u32 << bits)
    }

    /// `ec_dec_update` — narrow the range to the winning symbol and
    /// renormalize.
    pub fn dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng.wrapping_sub(s)
        };
        self.normalize();
    }

    /// `ec_dec_bit_logp` — decode a binary symbol whose "1" probability is
    /// `1/(1 << logp)`.
    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// `ec_dec_icdf` — decode a symbol via inverse-CDF table.
    /// `icdf[k] = ft - cumfreq[k+1]`, last entry must be 0. `ftb = log2(ft)`.
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        debug_assert!(!icdf.is_empty());
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret: usize = 0;
        let mut t;
        loop {
            t = s;
            s = r.wrapping_mul(icdf[ret] as u32);
            if d >= s {
                break;
            }
            ret += 1;
            if ret >= icdf.len() {
                ret -= 1;
                break;
            }
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// `ec_dec_uint` — uniform integer in `[0, ft)`. For ft > 256, the low
    /// bits are split off as raw bits.
    pub fn decode_uint(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 1);
        let ft_minus_1 = ft - 1;
        let ftb = 32 - ft_minus_1.leading_zeros();
        if ftb > EC_UINT_BITS {
            let ftb_extra = ftb - EC_UINT_BITS;
            let ft_top = (ft_minus_1 >> ftb_extra) + 1;
            let s = self.decode(ft_top);
            self.dec_update(s, s + 1, ft_top);
            let t = (s << ftb_extra) | self.decode_bits(ftb_extra);
            if t <= ft_minus_1 {
                t
            } else {
                self.error = true;
                ft_minus_1
            }
        } else {
            let s = self.decode(ft);
            self.dec_update(s, s + 1, ft);
            s
        }
    }

    /// `ec_dec_bits` — read raw bits from the back of the buffer.
    pub fn decode_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= (self.read_byte_from_end() as u32) << available;
                available += EC_SYM_BITS as i32;
                if available > EC_WINDOW_SIZE - EC_SYM_BITS as i32 {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits) - 1);
        self.end_window = window >> bits;
        self.nend_bits = available - bits as i32;
        self.nbits_total += bits as i32;
        ret
    }

    /// `ec_tell` — total bits consumed so far (whole bits).
    pub fn tell(&self) -> i32 {
        self.nbits_total - (32 - self.rng.leading_zeros()) as i32
    }

    /// `ec_tell_frac` — total bits consumed so far in 1/8 bit units.
    pub fn tell_frac(&self) -> u32 {
        // Match the lookup-based form in libopus entcode.c.
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

    pub fn error(&self) -> bool {
        self.error
    }

    pub fn total_bits(&self) -> u32 {
        self.storage * 8
    }

    /// Convenience aliases for the RFC pseudocode names.
    pub fn ec_dec_bits(&mut self, bits: u32) -> u32 {
        self.decode_bits(bits)
    }

    pub fn ec_dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        self.decode_icdf(icdf, ftb)
    }

    pub fn ec_dec_uint(&mut self, ft: u32) -> u32 {
        self.decode_uint(ft)
    }

    pub fn storage(&self) -> u32 {
        self.storage
    }
}

/// Fail-fast wrapper used at the top of a frame.
pub fn check_no_error(d: &RangeDecoder<'_>) -> Result<()> {
    if d.error() {
        Err(Error::invalid(
            "CELT range decoder: read past end of buffer",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_on_empty_buffer_does_not_panic() {
        let _d = RangeDecoder::new(&[]);
    }

    #[test]
    fn decode_bits_zero() {
        let mut d = RangeDecoder::new(&[0x01, 0x02, 0x03]);
        assert_eq!(d.decode_bits(0), 0);
    }

    #[test]
    fn decode_icdf_single_entry_returns_zero() {
        let mut d = RangeDecoder::new(&[0x80, 0x00, 0x00, 0x00]);
        let k = d.decode_icdf(&[0], 8);
        assert_eq!(k, 0);
    }

    #[test]
    fn decode_uint_in_range() {
        let mut d = RangeDecoder::new(&[0x55, 0xAA, 0x55, 0xAA]);
        let v = d.decode_uint(4);
        assert!(v < 4);
    }

    #[test]
    fn decode_uint_large_ft() {
        let mut d = RangeDecoder::new(&[0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA]);
        let v = d.decode_uint(1 << 16);
        assert!(v < (1 << 16));
    }

    #[test]
    fn tell_grows_as_bits_are_consumed() {
        let mut d = RangeDecoder::new(&[0x55, 0xAA, 0x55, 0xAA]);
        let t0 = d.tell();
        let _ = d.decode_bits(8);
        let t1 = d.tell();
        assert!(t1 >= t0);
    }
}
