//! Top-level Speex decoder factory.
//!
//! Extracts the 80-byte Speex header from `CodecParameters::extradata`
//! (which the Ogg demuxer fills with the first Speex packet), sanity-
//! checks it, and reports `Unsupported` — the synthesis loop (LSP
//! de-quantisation, pitch prediction, innovation codebook lookup, LPC
//! synthesis filter) is not yet implemented.

use oxideav_codec::Decoder;
use oxideav_core::{CodecParameters, Error, Result};

use crate::header::SpeexHeader;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.extradata.is_empty() {
        return Err(Error::invalid(
            "Speex decoder: missing extradata (expected Speex header packet)",
        ));
    }

    // Validating the header gives callers an accurate "this isn't Speex"
    // signal up-front, even though synthesis is unimplemented.
    let header = SpeexHeader::parse(&params.extradata)?;

    if header.nb_channels > 2 {
        return Err(Error::unsupported(format!(
            "Speex decoder: {}-channel stream",
            header.nb_channels
        )));
    }

    Err(Error::unsupported(format!(
        "Speex decoder is a scaffold — header OK (mode={:?}, rate={} Hz, channels={}), \
         CELP synthesis pending",
        header.mode, header.rate, header.nb_channels
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{SPEEX_HEADER_SIZE, SPEEX_SIGNATURE};
    use oxideav_core::CodecId;

    fn good_extradata() -> Vec<u8> {
        let mut h = vec![0u8; SPEEX_HEADER_SIZE];
        h[0..8].copy_from_slice(SPEEX_SIGNATURE);
        // version_id
        h[28..32].copy_from_slice(&1u32.to_le_bytes());
        h[32..36].copy_from_slice(&80u32.to_le_bytes());
        h[36..40].copy_from_slice(&8000u32.to_le_bytes());
        h[40..44].copy_from_slice(&0u32.to_le_bytes()); // NB
        h[48..52].copy_from_slice(&1u32.to_le_bytes()); // 1 channel
        h[52..56].copy_from_slice(&(-1i32).to_le_bytes());
        h[56..60].copy_from_slice(&160u32.to_le_bytes());
        h[64..68].copy_from_slice(&1u32.to_le_bytes());
        h
    }

    fn expect_err(params: &CodecParameters) -> Error {
        match make_decoder(params) {
            Ok(_) => panic!("expected make_decoder to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn empty_extradata_is_invalid() {
        let params = CodecParameters::audio(CodecId::new("speex"));
        assert!(matches!(expect_err(&params), Error::InvalidData(_)));
    }

    #[test]
    fn bad_signature_is_invalid() {
        let mut params = CodecParameters::audio(CodecId::new("speex"));
        params.extradata = vec![0u8; SPEEX_HEADER_SIZE];
        assert!(matches!(expect_err(&params), Error::InvalidData(_)));
    }

    #[test]
    fn good_header_returns_unsupported() {
        let mut params = CodecParameters::audio(CodecId::new("speex"));
        params.extradata = good_extradata();
        // Header parsed, but synthesis not implemented.
        assert!(matches!(expect_err(&params), Error::Unsupported(_)));
    }
}
