use std::collections::HashMap;

pub struct MarkovModel {
    pub frequencies: HashMap<Vec<char>,i32>,
    pub alphabet: Vec<char>,
}
impl MarkovModel {
    pub fn new() -> MarkovModel {
        MarkovModel{
            frequencies: HashMap::new(),
            alphabet: vec!()
        }
    }
    pub fn add_sequence(&mut self, sequence: &str) {
        let char_vec: Vec<char> = sequence.to_lowercase().chars().collect();
        for c in char_vec {
            // if vec!(c) is not already a new key, initialize with zero; then increment count
            *self.frequencies.entry(vec!(c)).or_insert(0) += 1;
        }
    }
}