//! MSB-first bit reader and §3.6 frame parser for G.729 packets.
//!
//! A G.729 frame is 80 bits (10 bytes) and is laid out MSB-first as
//! specified by Recommendation G.729 (January 2007) §3.6 / Table 8:
//!
//! ```text
//!   Field  Bits  Description
//!   L0      1    MA predictor switch (LSP)
//!   L1      7    first-stage LSP codebook index
//!   L2      5    second-stage low  LSP index
//!   L3      5    second-stage high LSP index
//!   P1      8    adaptive codebook delay, subframe 1
//!   P0      1    parity bit over the six MSBs of P1
//!   C1     13    fixed-codebook index,  subframe 1
//!   S1      4    fixed-codebook signs,  subframe 1
//!   GA1     3    gain-codebook stage 1, subframe 1
//!   GB1     4    gain-codebook stage 2, subframe 1
//!   P2      5    adaptive codebook delay, subframe 2 (relative)
//!   C2     13    fixed-codebook index,  subframe 2
//!   S2      4    fixed-codebook signs,  subframe 2
//!   GA2     3    gain-codebook stage 1, subframe 2
//!   GB2     4    gain-codebook stage 2, subframe 2
//!                ---------------------------------
//!   Total  80    bits = 10 bytes
//! ```

use oxideav_core::{Error, Result};

use crate::FRAME_BYTES;

/// Width in bits of each frame field, in transmission order.
pub const FIELD_WIDTHS: [u32; 15] = [1, 7, 5, 5, 8, 1, 13, 4, 3, 4, 5, 13, 4, 3, 4];

/// Parsed bit-fields of one G.729 frame (§3.6 / Table 8).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameParams {
    /// LSP MA-predictor switch.
    pub l0: u8,
    /// First-stage LSP codebook index.
    pub l1: u8,
    /// Second-stage low LSP codebook index.
    pub l2: u8,
    /// Second-stage high LSP codebook index.
    pub l3: u8,
    /// Subframe-1 adaptive-codebook (pitch) delay, 8-bit encoded.
    pub p1: u8,
    /// Parity bit over the six MSBs of `p1`.
    pub p0: u8,
    /// Subframe-1 algebraic fixed-codebook pulse positions (13 bits).
    pub c1: u16,
    /// Subframe-1 fixed-codebook pulse signs (4 bits).
    pub s1: u8,
    /// Subframe-1 gain codebook stage-1 index (3 bits).
    pub ga1: u8,
    /// Subframe-1 gain codebook stage-2 index (4 bits).
    pub gb1: u8,
    /// Subframe-2 adaptive-codebook delay, relative, 5 bits.
    pub p2: u8,
    /// Subframe-2 algebraic fixed-codebook pulse positions (13 bits).
    pub c2: u16,
    /// Subframe-2 fixed-codebook pulse signs (4 bits).
    pub s2: u8,
    /// Subframe-2 gain codebook stage-1 index (3 bits).
    pub ga2: u8,
    /// Subframe-2 gain codebook stage-2 index (4 bits).
    pub gb2: u8,
}

/// MSB-first bit reader for G.729 packets.
///
/// Bits are read most-significant-bit first within each byte, which is
/// how G.729 bits are serialised onto the wire.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    acc: u64,
    bits_in_acc: u32,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Total number of bits consumed so far.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_acc);
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `n` bits (0..=32) as an unsigned integer, MSB-first.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "G.729 BitReader::read_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("G.729 BitReader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read one bit as a bool.
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_u32(1)? != 0)
    }
}

/// Parse a 10-byte G.729 packet into [`FrameParams`].
pub fn parse_frame_params(packet: &[u8]) -> Result<FrameParams> {
    if packet.len() < FRAME_BYTES {
        return Err(Error::invalid(format!(
            "G.729 frame: expected {FRAME_BYTES} bytes, got {}",
            packet.len()
        )));
    }
    let mut br = BitReader::new(&packet[..FRAME_BYTES]);
    let l0 = br.read_u32(1)? as u8;
    let l1 = br.read_u32(7)? as u8;
    let l2 = br.read_u32(5)? as u8;
    let l3 = br.read_u32(5)? as u8;
    let p1 = br.read_u32(8)? as u8;
    let p0 = br.read_u32(1)? as u8;
    let c1 = br.read_u32(13)? as u16;
    let s1 = br.read_u32(4)? as u8;
    let ga1 = br.read_u32(3)? as u8;
    let gb1 = br.read_u32(4)? as u8;
    let p2 = br.read_u32(5)? as u8;
    let c2 = br.read_u32(13)? as u16;
    let s2 = br.read_u32(4)? as u8;
    let ga2 = br.read_u32(3)? as u8;
    let gb2 = br.read_u32(4)? as u8;
    // Post-condition: we must have consumed exactly 80 bits.
    debug_assert_eq!(br.bit_position(), 80);
    Ok(FrameParams {
        l0,
        l1,
        l2,
        l3,
        p1,
        p0,
        c1,
        s1,
        ga1,
        gb1,
        p2,
        c2,
        s2,
        ga2,
        gb2,
    })
}

/// Compute the expected parity bit (`P0`) for a given 8-bit pitch delay
/// `P1`. The parity is XOR of the six most-significant bits of `P1`
/// plus a leading 1, per G.729 §3.7.2.
pub fn pitch_parity(p1: u8) -> u8 {
    let mut x = (p1 >> 2) & 0x3F;
    x ^= 1;
    // XOR-fold to a single bit.
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    x & 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_reads_msb_first() {
        let mut br = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
        assert_eq!(br.read_u32(4).unwrap(), 0x5);
        assert_eq!(br.read_u32(8).unwrap(), 0xC3);
    }

    #[test]
    fn field_widths_sum_to_80() {
        let total: u32 = FIELD_WIDTHS.iter().sum();
        assert_eq!(total, 80);
    }

    #[test]
    fn parse_frame_params_known_pattern() {
        // Packet: all 1s. Every field should read as (1<<n)-1.
        let packet = [0xFFu8; 10];
        let p = parse_frame_params(&packet).unwrap();
        assert_eq!(p.l0, 0x1);
        assert_eq!(p.l1, 0x7F);
        assert_eq!(p.l2, 0x1F);
        assert_eq!(p.l3, 0x1F);
        assert_eq!(p.p1, 0xFF);
        assert_eq!(p.p0, 0x1);
        assert_eq!(p.c1, 0x1FFF);
        assert_eq!(p.s1, 0xF);
        assert_eq!(p.ga1, 0x7);
        assert_eq!(p.gb1, 0xF);
        assert_eq!(p.p2, 0x1F);
        assert_eq!(p.c2, 0x1FFF);
        assert_eq!(p.s2, 0xF);
        assert_eq!(p.ga2, 0x7);
        assert_eq!(p.gb2, 0xF);
    }

    #[test]
    fn parse_frame_params_rejects_short() {
        let packet = [0u8; 4];
        assert!(parse_frame_params(&packet).is_err());
    }

    #[test]
    fn pitch_parity_matches_spec_example() {
        // Parity of 0 is 1 (XOR of zero 1-bits plus the leading 1).
        assert_eq!(pitch_parity(0), 1);
        // Six 1-bits XOR 1 = (1^1^1^1^1^1) ^ 1 = 0 ^ 1 = 1.
        // MSBs of 0xFC are 0b111111 → six ones → XOR = 0, ^1 = 1.
        assert_eq!(pitch_parity(0xFC), 1);
        // MSBs of 0x7C (0b01111100) are 0b011111 → five ones → XOR = 1, ^1 = 0.
        assert_eq!(pitch_parity(0x7C), 0);
    }
}
