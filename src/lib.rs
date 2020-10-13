use std::collections::{HashMap, HashSet};
use std::cmp::max;

pub struct MarkovModel {
    pub frequencies: HashMap<Vec<char>,HashMap<char,i32>>,
    pub alphabet: HashSet<char>,
    order: i32
}
impl MarkovModel {

    const DEFAULT_ORDER: i32 = 3;

    pub fn new() -> MarkovModel {
        MarkovModel{
            frequencies: HashMap::new(),
            alphabet: HashSet::new(),
            order: MarkovModel::DEFAULT_ORDER,
        }
    }

    /// Takes in a vector of sequences (strings, for now), and calls the add_sequence function on
    /// each one in turn, training the model.
    ///
    /// ```
    /// use multimarkov::MarkovModel;
    /// let mut model = MarkovModel::new();
    /// let input_vec = vec!["a","foobar","baz"];
    /// assert!(model.add_sequences(input_vec).is_ok()); // assert short value "a" did not abort training
    /// assert!(model.frequencies.contains_key(&*vec!['b']));
    /// assert_eq!(*model.frequencies.get(&*vec!['b']).unwrap().get(&'a').unwrap(),2i32); // both sequences contain 'b' -> 'a' once
    /// ```
    pub fn add_sequences(&mut self, sequences: Vec<&str>) -> Result<(), &'static str> {
        if sequences.len() < 1 { return Err("no sequences in input"); }
        for sequence in sequences {
            match self.add_sequence(sequence) {
                Ok(()) => (),
                Err(e) => {
                    println!("error ignored: {}",e);
                }
            };
        }
        return Ok(());
    }

    /// Adds to the model all the observed state transitions found in one sequence of training data.
    /// This training is additive; it doesn't empty or overwrite the model, so you can call this
    /// method on many such training sequences in order to fully train the model.
    ///
    /// ```
    /// use multimarkov::MarkovModel;
    /// let mut model = MarkovModel::new();
    /// model.add_sequence("hello");
    /// assert!(model.frequencies.contains_key(&*vec!['l']));
    /// assert!(model.frequencies.contains_key(&*vec!['l','l']));
    /// assert!(model.frequencies.get(&*vec!['l']).unwrap().contains_key(&'l'));
    /// assert!(model.frequencies.get(&*vec!['l','l']).unwrap().contains_key(&'o'));
    /// ```
    pub fn add_sequence(&mut self, sequence: &str) -> Result<(), &'static str> {
        if sequence.len() < 2 { return Err("sequence was too short, must contain at least two characters"); }

        let char_vec: Vec<char> = sequence.to_lowercase().chars().collect();
        // loop backwards through the characters in the sequence
        for i in (1..char_vec.len()).rev() {
            // Build a running set of all known characters while we're at it
            self.alphabet.insert(char_vec[i]);
            // For the sequences preceding character (i), record that character (i) was observed following them.
            // IE if the char_vec is ['R','U','S','T'] and this is a 3rd-order model, then for the three models ['S'], ['U','S'], and ['R','U','S'] we record that ['T'] is a known follower.
            for j in (max(0,i as i32 - self.order) as usize)..i {
                *self.frequencies.entry(Vec::from(&char_vec[j..i])).or_insert(HashMap::new()).entry(char_vec[i]).or_insert(0) += 1;
            }
        }
        self.alphabet.insert(char_vec[0]); // previous loop stops before index 0
        Ok(())
    }

}