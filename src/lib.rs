pub mod builder;

use crate::builder::MultiMarkovBuilder;
use rand::{Rng, RngCore};
use std::cmp::min;
use std::collections::{BTreeMap, HashMap, HashSet};
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
/// let mm = MultiMarkov::<char>::builder()
///     .with_order(2) // omit to use default of 3
///     .with_prior(0.01) // omit to use default of 0.005, or call .without_prior() to disable priors
///     .train(input_vec.into_iter())
///     .build();
/// ```
///
/// Use method `random_next` (see below) to use it to generate new sequences.
pub struct MultiMarkov<T, R>
where
    T: Eq + Hash + Clone + std::cmp::Ord,
    R: RngCore,
{
    pub markov_chain: HashMap<Vec<T>, BTreeMap<T, f64>>,
    pub known_states: HashSet<T>,
    pub order: i32,
    pub rng: R,
}

impl<T, R> MultiMarkov<T, R>
where
    T: Eq + Hash + Clone + std::cmp::Ord,
    R: RngCore,
{
    pub const DEFAULT_ORDER: i32 = 3;
    pub const DEFAULT_PRIOR: f64 = 0.005;

    /// Create a builder to set up and train a MultiMarkov instance.
    pub fn builder(rng: R) -> MultiMarkovBuilder<T, R> {
        MultiMarkovBuilder::<T, R>::new(rng)
    }

    /// Using the random-number generator and the "weights" of the various state transitions from
    /// the trained model, draw a new state to follow the given sequence.
    pub fn random_next(&mut self, current_sequence: &Vec<T>) -> Option<T> {
        let r: f64 = self.rng.gen();
        let bestmodel = self.best_model(current_sequence)?;
        let sum_of_weights: f64 = bestmodel.values().sum();
        let mut randomroll = r * sum_of_weights; // TODO: can this be accomplished in fewer lines?
                                                 // every state has a chance of being selected in proportion to its 'weight' as fraction of the sum of weights
        for (k, v) in bestmodel {
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
    fn best_model(&self, current_sequence: &Vec<T>) -> Option<&BTreeMap<T, f64>> {
        // If current_sequence.len() is at least self.order, count "i" down from self.order to 1,
        // taking sequence slices of length "i" and checking if we have a matching model:
        for i in (1..(min(self.order as usize, current_sequence.len()) + 1)).rev() {
            let subsequence =
                &current_sequence[(current_sequence.len() - i)..current_sequence.len()];
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
    use rand::{rngs::ThreadRng, thread_rng};

    fn char_data() -> Vec<Vec<char>> {
        vec![
            vec!['a'], // can't be used, but should be skipped over rather than causing error to propagate
            vec!['a', 'c', 'e'],
            vec!['f', 'o', 'o', 'b', 'a', 'r'],
            vec!['b', 'a', 'z'],
        ]
    }

    #[test]
    fn test_model_builder_works() {
        let mut mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .with_prior(0.015)
            .train(char_data().into_iter())
            .build();
        assert!(mm.random_next(&vec!['a', 'b', 'c']).is_some()); // random draw didn't fail (because 'c' is in training data)
        assert!(mm.random_next(&vec!['x', 'y', 'z']).is_none()); // 'z' is in training data only at end of sequence; no following states were observed so there's no model
    }

    #[test]
    fn test_model_weights_and_priors_are_correct() {
        let mut mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .with_prior(0.001)
            .train(char_data().into_iter())
            .build();
        let chain = &mm.markov_chain;
        assert_eq!(*chain.get(&*vec!['b']).unwrap().get(&'a').unwrap(), 2.0); // seen twice in training data
        assert_eq!(*chain.get(&*vec!['a']).unwrap().get(&'c').unwrap(), 1.0); // seen once in training data
        assert_eq!(*chain.get(&*vec!['a']).unwrap().get(&'e').unwrap(), 0.001); // not observed in training data; assigned a 'prior' probability
    }
}
