use std::collections::{HashMap, HashSet};

pub struct MarkovModel {
    pub frequencies: HashMap<Vec<char>,HashMap<Vec<char>,i32>>,
    pub alphabet: HashSet<char>,
}
impl MarkovModel {
    pub fn new() -> MarkovModel {
        MarkovModel{
            frequencies: HashMap::new(),
            alphabet: HashSet::new()
        }
    }
    pub fn add_sequence(&mut self, sequence: &str) -> Result<(), &'static str> {
        if sequence.len() < 2 { return Err("sequence was too short, must contain at least two characters") };

        let char_vec: Vec<char> = sequence.to_lowercase().chars().collect();
        // loop backwards through the characters in the sequence
        for i in (1..char_vec.len()).rev() {
            // Build a running set of all known characters while we're at it
            self.alphabet.insert(char_vec[i]);
            // For the character at (i-1), record that character at (i) occurred following it.
            // TODO: build multi-order models
            *self.frequencies.entry(vec!(char_vec[i-1])).or_insert(HashMap::new()).entry(vec!(char_vec[i])).or_insert(0) += 1;
            println!("{}",i);
        }
        self.alphabet.insert(char_vec[0]); // previous loop stops before index 0
        Ok(())
    }
}