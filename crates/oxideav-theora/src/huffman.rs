//! Theora Huffman tree decode.
//!
//! Theora stores 80 Huffman trees in its setup header (spec §6.4.4). Each
//! tree is serialised via a depth-first traversal where:
//!
//! ```text
//!   bit 0 = "internal node" → recurse left (0), recurse right (1)
//!   bit 1 = "leaf"           → then read 5-bit token value (0..=31)
//! ```
//!
//! (Note: this is the OPPOSITE of a few other codec Huffman encodings —
//! Theora's "1 = leaf" matches the libtheora reference.)
//!
//! Tree depth is capped at 32 bits per codeword (spec §6.4.4) and at most 32
//! leaves per tree.
//!
//! We represent trees as a compact array of nodes. Node 0 is the root; each
//! node is either a leaf (carrying a token) or an internal pair `(left,
//! right)` referencing other node indices.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

#[derive(Clone, Copy, Debug)]
pub enum Node {
    Leaf(u8),
    Inner { left: u16, right: u16 },
}

#[derive(Clone, Debug, Default)]
pub struct HuffmanTree {
    pub nodes: Vec<Node>,
}

impl HuffmanTree {
    pub fn new_empty() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Build a tree by reading from `br`. Enforces max depth 32 per
    /// Theora spec §6.4.4 (codewords cannot exceed 32 bits).
    pub fn read_from(br: &mut BitReader<'_>) -> Result<Self> {
        let mut t = HuffmanTree {
            nodes: Vec::with_capacity(64),
        };
        let root = t.build_node(br, 0)?;
        debug_assert_eq!(root, 0);
        Ok(t)
    }

    fn build_node(&mut self, br: &mut BitReader<'_>, depth: u32) -> Result<u16> {
        if depth > 32 {
            return Err(Error::invalid("Theora Huffman tree exceeds depth 32"));
        }
        let idx = self.nodes.len();
        if idx >= u16::MAX as usize {
            return Err(Error::invalid("Theora Huffman tree too large"));
        }
        // Push a placeholder so child nodes can take subsequent indices
        // without stomping on this slot.
        self.nodes.push(Node::Leaf(0));
        // Theora: bit 1 = leaf, bit 0 = internal node.
        let is_leaf = br.read_bit()?;
        if is_leaf {
            let tok = br.read_u32(5)? as u8;
            self.nodes[idx] = Node::Leaf(tok);
        } else {
            let left = self.build_node(br, depth + 1)?;
            let right = self.build_node(br, depth + 1)?;
            self.nodes[idx] = Node::Inner { left, right };
        }
        Ok(idx as u16)
    }

    /// Decode one token by walking the tree. Bit 0 goes left, bit 1 right
    /// (matches the build order above).
    pub fn decode(&self, br: &mut BitReader<'_>) -> Result<u8> {
        if self.nodes.is_empty() {
            return Err(Error::invalid("empty Theora Huffman tree"));
        }
        let mut idx = 0usize;
        for _ in 0..64 {
            match self.nodes[idx] {
                Node::Leaf(v) => return Ok(v),
                Node::Inner { left, right } => {
                    let bit = br.read_bit()?;
                    idx = if bit { right as usize } else { left as usize };
                }
            }
        }
        Err(Error::invalid("Theora Huffman walk exceeded 64 iterations"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_leaf_tree() {
        // A tree with only a root leaf is valid (1 token).
        // Bit layout: leaf(1) + 5-bit tok 7 (00111)
        // Bits: 1 00111 00  → 0b10011100 = 0x9C
        let bits = [0b10011100u8];
        let mut br = BitReader::new(&bits);
        let t = HuffmanTree::read_from(&mut br).unwrap();
        assert_eq!(t.nodes.len(), 1);
        let mut br2 = BitReader::new(&[0u8]);
        assert_eq!(t.decode(&mut br2).unwrap(), 7);
    }

    #[test]
    fn two_leaf_tree() {
        // Internal(root) = 0; left = leaf(1) tok 5 = 1 00101; right = leaf(1) tok 10 = 1 01010
        // Bits: 0 1 00101 1 01010  (14 bits)
        //       0 10010 11 010 10
        //       01001011 01010_ _ _
        //       0x4B       0x50
        let bits = [0b01001011u8, 0b01010000u8];
        let mut br = BitReader::new(&bits);
        let t = HuffmanTree::read_from(&mut br).unwrap();
        assert_eq!(t.nodes.len(), 3);
        // Decode "0" → token 5.
        let mut br_l = BitReader::new(&[0u8]);
        assert_eq!(t.decode(&mut br_l).unwrap(), 5);
        // Decode "1" → token 10.
        let mut br_r = BitReader::new(&[0x80u8]);
        assert_eq!(t.decode(&mut br_r).unwrap(), 10);
    }
}
