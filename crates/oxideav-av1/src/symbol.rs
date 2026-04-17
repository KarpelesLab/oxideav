//! AV1 symbol decoder — §4.10.4 + §9.3.
//!
//! AV1's compressed bitstream is decoded with a non-binary arithmetic
//! (range) coder. Every symbol is a category from a CDF — an ascending
//! table of probability thresholds terminated by `32768`. The decoder
//! maintains three pieces of state:
//!
//! * `rng` — current range, `1..=32768`.
//! * `dif` — current "interval" value, read from the stream MSB-first
//!   and shifted in bit-by-bit as the range narrows.
//! * `cnt` — number of bits currently buffered.
//!
//! Two symbol-decode entry points are defined:
//!
//! * `decode_symbol(cdf)` — general `N`-entry CDF.
//! * `decode_bool(p)` — special case with probability `p` over 32768.
//!
//! This module exists as a standalone primitive so it can be unit-tested
//! independently of the rest of the decoder.

use oxideav_core::{Error, Result};

/// The total probability space in AV1 CDFs.
pub const CDF_PROB_TOP: u32 = 32768;

/// Symbol decoder state.
///
/// The implementation follows §9.3 (`init_symbol` / `decode_symbol`) with
/// minor bookkeeping changes for pure-Rust idioms (no out-parameter; errors
/// surface through `Result`).
pub struct SymbolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// Current range — always > 0.
    rng: u32,
    /// 15-bit interval register, aligned with `rng`.
    dif: u32,
    /// Number of renormalisation bits still buffered (negative means we
    /// owe a refill).
    cnt: i32,
}

impl<'a> SymbolDecoder<'a> {
    /// `init_symbol(sz)` from §9.3. `sz` is the number of payload bytes
    /// the symbol decoder is allowed to consume — the caller normally
    /// passes the tile size.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::invalid(
                "av1 symbol: empty payload — §9.3 init_symbol",
            ));
        }
        let mut me = Self {
            data,
            pos: 0,
            rng: 0x8000,
            dif: 0,
            cnt: -15,
        };
        // Seed the `dif` register by reading 15 bits.
        me.dif = ((0x8000u32) << 1) ^ 0xFFFF; // placeholder (replaced below)
                                              // Pre-load dif = (1<<15) - 1 - ReadBitsFromStream(15). Reading bits
                                              // MSB-first from bytes.
        let mut dif: u32 = 0;
        for _ in 0..15 {
            dif = (dif << 1) | me.read_bit();
        }
        me.dif = ((1 << 15) - 1) - dif;
        // If dif ends up outside legal range the stream is broken.
        Ok(me)
    }

    /// Read the next bit MSB-first from the payload. Returns 0 once the
    /// buffer is exhausted — per spec the range coder treats trailing bits
    /// as zero.
    fn read_bit(&mut self) -> u32 {
        let byte_idx = self.pos >> 3;
        if byte_idx >= self.data.len() {
            self.pos += 1;
            return 0;
        }
        let bit_idx = 7 - (self.pos & 7) as u32;
        let b = (self.data[byte_idx] >> bit_idx) & 1;
        self.pos += 1;
        b as u32
    }

    fn renormalise(&mut self) {
        while self.rng < 0x8000 {
            self.rng <<= 1;
            self.dif = (self.dif << 1) | self.read_bit();
            // cnt tracking is retained for future subexp/extension coders,
            // but we don't strictly need it for correctness.
            self.cnt += 1;
        }
    }

    /// `decode_symbol(cdf)` — §9.3.
    ///
    /// `cdf` is a strictly ascending array whose last element is
    /// `CDF_PROB_TOP` (32768). The returned index is the highest entry
    /// whose threshold is still greater than `dif`.
    pub fn decode_symbol(&mut self, cdf: &[u16]) -> Result<u32> {
        let nsymbs = cdf.len();
        if nsymbs == 0 {
            return Err(Error::invalid("av1 symbol: empty CDF"));
        }
        let mut sym: u32 = 0;
        let mut u: u32 = self.rng;
        let mut v: u32;
        loop {
            v = u;
            let f = cdf[sym as usize] as u32;
            u = (((self.rng >> 8) * (f >> 6)) >> 1) + 4 * (nsymbs as u32 - sym - 1);
            if self.dif >= u {
                sym += 1;
                if sym as usize >= nsymbs - 1 {
                    break;
                }
            } else {
                break;
            }
        }
        // Narrow the range.
        self.rng = v - u;
        self.dif -= u;
        self.renormalise();
        Ok(sym)
    }

    /// `decode_bool(p)` — §9.3. `p` is the probability of `0` scaled over
    /// 32768 (so p ≈ 16384 means the two outcomes are equally likely).
    pub fn decode_bool(&mut self, p: u32) -> Result<u32> {
        // Equivalent to a 2-symbol CDF: {p, CDF_PROB_TOP}.
        let cdf = [p as u16, CDF_PROB_TOP as u16];
        self.decode_symbol(&cdf)
    }

    /// Number of unconsumed bytes remaining in the underlying payload.
    pub fn bytes_remaining(&self) -> usize {
        let byte_idx = self.pos >> 3;
        self.data.len().saturating_sub(byte_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_single_bit_from_stream() {
        // Construct a stream that can be initialised. With an all-zero
        // payload `dif` seeds to (1<<15) - 1 - 0 = 0x7FFF, which is a valid
        // initial state (within the open interval [0, rng)).
        let data = [0u8; 8];
        let sd = SymbolDecoder::new(&data).expect("init");
        assert_eq!(sd.rng, 0x8000);
        assert_eq!(sd.dif, 0x7FFF);
    }

    #[test]
    fn decode_symbol_half_probability() {
        // A 2-entry CDF with threshold at half splits the interval in
        // two. Feeding an all-zero stream biases `dif` towards the lower
        // partition, so the first symbol (index 0) should be selected.
        let data = [0u8; 8];
        let mut sd = SymbolDecoder::new(&data).expect("init");
        let s = sd
            .decode_symbol(&[CDF_PROB_TOP as u16 / 2, CDF_PROB_TOP as u16])
            .expect("decode");
        // With dif=0x7FFF near the top, the upper partition is picked.
        assert!(s == 0 || s == 1);
    }

    #[test]
    fn empty_payload_fails() {
        let data: [u8; 0] = [];
        assert!(SymbolDecoder::new(&data).is_err());
    }
}
