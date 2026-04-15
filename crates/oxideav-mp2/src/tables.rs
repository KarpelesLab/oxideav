//! MPEG-1 Audio Layer II bit-allocation tables — placeholder.
//!
//! Real tables are B.2a–d in ISO/IEC 11172-3 §2.4. Populated in a
//! follow-up session; the stub below only exists so the decoder
//! scaffold compiles.

pub struct AllocTable {
    pub sblimit: usize,
    pub nbal: &'static [u8],
}
