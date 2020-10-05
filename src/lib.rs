use std::collections::{HashMap, HashSet};

pub struct MarkovModel {
    pub frequencies: HashMap<Vec<char>,i32>,
    pub alphabet: HashSet<char>,
}
impl MarkovModel {
    pub fn new() -> MarkovModel {
        MarkovModel{
            frequencies: HashMap::new(),
            alphabet: HashSet::new()
        }
    }
    pub fn add_sequence(&mut self, sequence: &str) {
        let char_vec: Vec<char> = sequence.to_lowercase().chars().collect();
        for c in char_vec {
            // Build a running count of observances of each character:
            // if vec!(c) is not already a key, initialize with zero; then increment count
            *self.frequencies.entry(vec!(c)).or_insert(0) += 1;
            // Build a running set of all known characters while we're at it
            self.alphabet.insert(c);
        }
    }
}