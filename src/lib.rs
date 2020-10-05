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
            if let Some(x) = self.frequencies.get_mut(&vec!(c)) {
                *x = *x + 1;
            } else {
                self.frequencies.insert(vec!(c),1);
            }
        }
    }
}