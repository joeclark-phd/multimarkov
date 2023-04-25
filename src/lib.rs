use std::collections::{HashMap, HashSet};
use std::cmp::{max,min};
use rand::Rng;
use std::hash::Hash;

/// Multi-order Markov chain models with a Katz back-off, for procedural generation applications.
///
/// A Markov chain maps current states to possible future states, usually providing probabilities
/// for each possible state transition ([c.f. Wikipedia](https://en.wikipedia.org/wiki/Markov_chain)).
/// This is useful in procedural generation, for example to model which letters in a language most
/// frequently follow a given letter, and to randomly generate a sequence of future states based on
/// the probabilities quantified in the chain.
///
/// This struct offers multi-order Markov chain models with a Katz back-off.  What that means is,
/// if `order > 1`, multiple models of varying fittedness may specify possible following states
/// for a given sequence.  For example, if `order == 3` (the default) and you have a sequence
/// `['R','U','S','T']`, for which you'd like to randomly draw future states, your instance of
/// MultiMarkov may have a model for states that follow `['U','S','T']`, another for states
/// that follow `['S','T']`, and a third for states that follow `['T']`.  These models are built up
/// by ingesting vectors of training data (for example, text from a file).  The "Katz back-off"
/// means that in drawing random future states we try to use the probability distribution in the
/// most tightly-fitted possible model (the one for `['U','S','T']`), but if no such model exists
/// (i.e., if the sequence `['U','S','T']` was never seen in the training data with any following
/// state), we then "back off" to the next-best-fitted model (`['S','T']`) and so on, until we find
/// a trained model.  A model will certainly be found if `['T']` was even once observed in the
/// training data with a following state.
///
/// A feature that may be desired in procedural generation applications is the option to inject some
/// "true randomness" in the form of "Dirichlet prior" relative probabilities, i.e., small weights
/// given to state transitions *not* observed in training data.  These can make up for the sparsity
/// of a training dataset and enable the generation of sequences not observed in training.
///
/// This implementation is inspired by the algorithm
/// [described by JLund3 at RogueBasin](http://www.roguebasin.com/index.php/Names_from_a_high_order_Markov_Process_and_a_simplified_Katz_back-off_scheme).
///
/// Instantiate it with the builder pattern:
///
/// ```
/// use multimarkov::MultiMarkov;
/// let input_vec = vec![
///     vec!['a','c','e'],
///     vec!['f','o','o','b','a','r'],
///     vec!['b','a','z'],
/// ];
/// let markov = MultiMarkov::<char>::new()
///     .with_order(2) // omit to use default of 3
///     .with_priors(0.01) // omit to use default of 0.005
///     .train(input_vec.into_iter())
///     .build();
/// ```
///
/// Use method `random_next` (see below) to use it to generate new sequences.
pub struct MultiMarkov<T: Eq + Hash + Clone + Copy> {
    pub markov_chain: HashMap<Vec<T>,HashMap<T,f64>>,
    pub known_states: HashSet<T>,
    order: i32,
    prior: f64,
    // TODO: add a random number generator (or seed?) that the user can specify, or go with a default
}
impl<T: Eq + Hash + Clone + Copy> MultiMarkov<T> {

    pub const DEFAULT_ORDER: i32 = 3;
    pub const DEFAULT_PRIOR: f64 = 0.005;

    pub fn new() -> MultiMarkov<T> {
        MultiMarkov {
            markov_chain: HashMap::new(),
            known_states: HashSet::new(),
            order: MultiMarkov::<T>::DEFAULT_ORDER,
            prior: MultiMarkov::<T>::DEFAULT_PRIOR,
        }
    }

    pub fn with_order(mut self, order: i32) -> Self {
        self.order = order;
        self
    }

    pub fn with_priors(mut self, prior: f64) -> Self {
        self.prior = prior;
        self
    }

    pub fn without_priors(mut self) -> Self {
        self.prior = 0.0;
        self
    }

    /// Ingest an iterator of sequences, adding the observed state transitions to the internal
    /// statistical model.
    pub fn train(mut self, sequences: impl Iterator<Item = Vec<T>>) -> Self {
        self.add_sequences(sequences);
        self
    }

    /// Takes in a vector of sequences, and calls the `add_sequence` method on
    /// each one in turn, training the model.
    fn add_sequences(&mut self, sequences: impl Iterator<Item = Vec<T>>) -> Result<(), &'static str> {
        //if sequences.len() < 1 { return Err("no sequences in input"); }
        let mut sequence_count: usize = 0;
        for sequence in sequences {
            match self.add_sequence(&sequence) {
                Ok(()) => sequence_count+=1,
                Err(e) => {
                    println!("error ignored: {}",e);
                }
            };
        }
        println!("{} sequences added",sequence_count);
        return Ok(());
    }

    /// Adds to the model all the observed state transitions found in one sequence of training data.
    /// This training is additive; it doesn't empty or overwrite the model, so you can call this
    /// method on many such training sequences in order to fully train the model.
    fn add_sequence(&mut self, sequence: &Vec<T>) -> Result<(), &'static str> {
        if sequence.len() < 2 { return Err("sequence was too short, must contain at least two states"); }

        // loop backwards through the characters in the sequence
        for i in (1..sequence.len()).rev() {
            // Build a running set of all known characters while we're at it
            self.known_states.insert(sequence[i]);
            // For the sequences preceding character (i), record that character (i) was observed following them.
            // IE if the char_vec is ['R','U','S','T'] and this is a 3rd-order model, then for the three models ['S'], ['U','S'], and ['R','U','S'] we record that ['T'] is a known follower.
            for j in (max(0,i as i32 - self.order) as usize)..i {
                *self.markov_chain.entry(Vec::from(&sequence[j..i])).or_insert(HashMap::new()).entry(sequence[i]).or_insert(0.0) += 1.0;
            }
        }
        self.known_states.insert(sequence[0]); // previous loop stops before index 0
        Ok(())
    }

    /// As the final step, we add priors (or "prior probabilities").  The model is now fully built.
    pub fn build(mut self) -> Self {
        self.add_priors(self.prior);
        self
    }

    /// Fills in missing state transitions with a given value so that any observed state (except
    /// those only seen at the end of sequences) can transition to any other state.
    fn add_priors(&mut self, prior: f64) {
        for v in self.markov_chain.values_mut() {
            for &a in self.known_states.iter() {
                v.entry(a).or_insert(prior);
            }
        }
    }

    /// Using the random-number generator and the "weights" of the various state transitions from
    /// the trained model, draw a new state to follow the given sequence.
    pub fn random_next(&self, current_sequence: &Vec<T>) -> Option<T> {
        let bestmodel = self.best_model(current_sequence)?;
        let sum_of_weights: f64 = bestmodel.values().sum();
        // TODO: use an RNG or RNG seed stored in the struct, so the user can specify it if desired
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut randomroll = r*sum_of_weights; // TODO: can this be accomplished in fewer lines?
        // every state has a chance of being selected in proportion to its 'weight' as fraction of the sum of weights
        for (k,v) in bestmodel {
            if randomroll > *v {
                randomroll -= v;
            } else {
                return Some(k.clone());
            }
        }
        None // this should never be reached
    }

    /// For a given sequence, find the most tightly-fitted model we have for its tail-end subsequence.
    /// For example, if the sequence is `['t','r','u','s']`, and self.order==3, first see if we have
    /// a model for `['r','u','s']`, which will only exist if that sequence has been seen in the training
    /// data.  If not, see if we have a model for `['u','s']`, and failing that, see if we have a
    /// model for `['s']`.  If no model for `['s']` is found, return `None`.
    fn best_model(&self, current_sequence: &Vec<T>) ->  Option<&HashMap<T,f64>> {
        // If current_sequence.len() is at least self.order, count "i" down from self.order to 1,
        // taking sequence slices of length "i" and checking if we have a matching model:
        for i in (1..(min(self.order as usize, current_sequence.len())+1)).rev() {
            let subsequence = &current_sequence[(current_sequence.len()-i)..current_sequence.len()];
            if self.markov_chain.contains_key(subsequence) {
                return self.markov_chain.get(subsequence);
            }
        }
        None
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_builder_works() {
        let input_vec = vec![
            vec!['a'], // can't be used, but should be skipped over rather than causing error to propagate
            vec!['a','c','e'],
            vec!['f','o','o','b','a','r'],
            vec!['b','a','z'],
        ];
        let mut markov = MultiMarkov::<char>::new()
            .with_order(MultiMarkov::<char>::DEFAULT_ORDER)
            .with_priors(MultiMarkov::<char>::DEFAULT_PRIOR)
            .train(input_vec.into_iter())
            .build();
        assert!(markov.random_next(&vec!['a','b','c']).is_some()); // random draw didn't fail (because 'c' is in training data)
        assert!(markov.random_next(&vec!['x','y','z']).is_none()); // 'z' is in training data only at end of sequence; no following states were observed so there's no model
    }

    #[test]
    fn test_model_weights_and_priors_are_correct() {
        let input_vec = vec![
            vec!['a','b'],
            vec!['a','b','c'],
        ];
        let markov = MultiMarkov::<char>::new()
            .with_priors(0.001)
            .train(input_vec.into_iter())
            .build();
        let chain = &markov.markov_chain;
        assert_eq!(*chain.get(&*vec!['a']).unwrap().get(&'b').unwrap(),2.0); // seen twice in training data
        assert_eq!(*chain.get(&*vec!['b']).unwrap().get(&'c').unwrap(),1.0); // seen once in training data
        assert_eq!(*chain.get(&*vec!['b']).unwrap().get(&'a').unwrap(),0.001); // not observed in training data; assigned a 'prior' probability
    }

}

