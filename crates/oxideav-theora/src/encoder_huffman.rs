//! Encoder-side Huffman lookup table builder.
//!
//! The decoder represents Huffman trees as walked nodes
//! ([`crate::huffman::HuffmanTree`]). For encoding we need the inverse: for
//! each token (0..=31), the bit code and its length. This module walks each
//! tree and extracts `(code, length)` pairs, returning a vector of 32-entry
//! tables (one per tree).
//!
//! Per-token codes follow the same MSB-first convention as decode: a left edge
//! contributes a `0` bit, a right edge a `1` bit. The most-significant bit of
//! the code is the first bit of the codeword.
//!
//! Tokens absent from a tree have length 0 (cannot be encoded).

use crate::huffman::{HuffmanTree, Node};

#[derive(Clone, Copy, Debug, Default)]
pub struct HuffCode {
    pub code: u32,
    pub len: u8,
}

/// Per-tree encode table — one slot per token id 0..=31.
pub type HuffTable = [HuffCode; 32];

pub fn build_encode_table(tree: &HuffmanTree) -> HuffTable {
    let mut out = [HuffCode::default(); 32];
    if tree.nodes.is_empty() {
        return out;
    }
    walk(tree, 0, 0, 0, &mut out);
    out
}

fn walk(tree: &HuffmanTree, node: usize, code: u32, depth: u8, out: &mut HuffTable) {
    match tree.nodes[node] {
        Node::Leaf(tok) => {
            // A leaf at depth=0 (single-leaf tree) produces a 0-length code.
            // That cannot actually be emitted; mark as length 1 with code 0
            // so callers don't trip on a zero-length write. (Such trees can
            // only encode one token, and the decoder would never read a bit.)
            let len = depth.max(1);
            let code = if depth == 0 { 0 } else { code };
            if (tok as usize) < out.len() {
                out[tok as usize] = HuffCode { code, len };
            }
        }
        Node::Inner { left, right } => {
            if depth >= 32 {
                return;
            }
            walk(tree, left as usize, code << 1, depth + 1, out);
            walk(tree, right as usize, (code << 1) | 1, depth + 1, out);
        }
    }
}

/// Build encode tables for all 80 trees in the parsed setup.
pub fn build_all(huffs: &[HuffmanTree]) -> Vec<HuffTable> {
    huffs.iter().map(build_encode_table).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn two_leaf_tree_round_trip() {
        // Tree with two leaves: left = token 5, right = token 10.
        // Build from the same bit stream as huffman.rs's two_leaf_tree test.
        let bits = [0b01001011u8, 0b01010000u8];
        let mut br = BitReader::new(&bits);
        let t = HuffmanTree::read_from(&mut br).unwrap();
        let table = build_encode_table(&t);
        assert_eq!(table[5].len, 1);
        assert_eq!(table[5].code, 0);
        assert_eq!(table[10].len, 1);
        assert_eq!(table[10].code, 1);
    }
}
