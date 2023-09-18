use crate::MultiMarkov;
use rand::RngCore;
use std::cmp::max;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;

pub struct MultiMarkovBuilder<T, R>
where
    T: Eq + Hash + Clone + std::cmp::Ord,
    R: RngCore,
{
    pub markov_chain: HashMap<Vec<T>, BTreeMap<T, f64>>,
    pub known_states: HashSet<T>,
    order: i32,
    prior: Option<f64>,
    rng: R,
}

impl<T, R> MultiMarkovBuilder<T, R>
where
    T: Eq + Hash + Clone + std::cmp::Ord,
    R: RngCore,
{
    /// Instantiate a new builder.
    pub fn new(rng: R) -> Self {
        Self {
            markov_chain: HashMap::new(),
            known_states: HashSet::new(),
            order: MultiMarkov::<T, R>::DEFAULT_ORDER,
            prior: Some(MultiMarkov::<T, R>::DEFAULT_PRIOR),
            rng,
        }
    }

    /// Specify the "order" of the Markov model.  Must be a positive integer.
    /// We recommend small values from about 1 to 3.  Higher values will make the procedurally
    /// generated data more similar to the training data, less random, and will make the process
    /// slower and require more memory.
    ///
    /// The default is MultiMarkov::DEFAULT_ORDER
    pub fn with_order(mut self, order: i32) -> Self {
        assert!(order > 0, "Order must be an integer greater than zero.");
        self.order = order;
        self
    }

    /// Specifies the "prior probability" of transition from any known state to any other known state,
    /// if that transition was not observed in the training data.  Small fractions are recommended,
    /// so that this "true randomness" will be less common than transitions based on the training data.
    ///
    /// The default is MultiMarkov::DEFAULT_PRIOR
    pub fn with_prior(mut self, prior: f64) -> Self {
        if prior == 0.0 {
            self.prior = None;
        } else {
            self.prior = Some(prior);
        }
        self
    }

    /// Specifies that there will be no use of "prior probability" in this model.  The only state
    /// transitions possible will be those seen in the training data.
    pub fn without_prior(mut self) -> Self {
        self.prior = None;
        self
    }

    /// Ingest an iterator of sequences, adding the observed state transitions to the internal
    /// statistical model.
    pub fn train(mut self, sequences: impl Iterator<Item = Vec<T>>) -> Self {
        let mut success_count: usize = 0;
        let mut error_count: usize = 0;
        for sequence in sequences {
            match self.train_sequence(sequence) {
                Ok(()) => success_count += 1,
                Err(_) => error_count += 1,
            };
        }
        println!(
            "{} sequences successfully trained; {} errors",
            success_count, error_count
        );
        self
    }

    /// Learn all the transitions possible from one training sequence, adding observations to the Markov model.
    fn train_sequence(&mut self, sequence: Vec<T>) -> Result<(), &str> {
        if sequence.len() < 2 {
            return Err("sequence was too short, must contain at least two states");
        }

        // loop backwards through the characters in the sequence
        for i in (1..sequence.len()).rev() {
            // Build a running set of all known characters while we're at it
            self.known_states.insert(sequence[i].clone());

            // For the sequences preceding character (i), record that character (i) was observed following them.
            // IE if the char_vec is ['R','U','S','T'] and this is a 3rd-order model, then for the three models ['S'], ['U','S'], and ['R','U','S'] we record that ['T'] is a known follower.
            for j in (max(0, i as i32 - self.order) as usize)..i {
                if let Some(transitions_from) = self.markov_chain.get_mut(&sequence[j..i]) {
                    // "from" sequence has been seen before
                    if let Some(weight) = transitions_from.get_mut(&sequence[i]) {
                        // it has been seen before with this transition; add one observance
                        *weight += 1.0;
                    } else {
                        // it hasn't been seen before with this transition; insert transition with one observance
                        transitions_from.insert(sequence[i].clone(), 1.0);
                    }
                } else {
                    // "from" sequence hasn't been seen before; add it and add the observed transition
                    let mut observed_transition = BTreeMap::new();
                    observed_transition.insert(sequence[i].clone(), 1.0);
                    self.markov_chain
                        .insert(Vec::from(&sequence[j..i]), observed_transition);
                }
                // The following one-liner might accomplish all of the above, but is pretty hard on the eyes:
                //     *self.markov_chain.entry(Vec::from(&sequence[j..i])).or_insert(HashMap::new()).entry(sequence[i].clone()).or_insert(0.0) += 1.0;
            }
        }

        Ok(())
    }

    /// Adds prior probabilities (if any) and builds the MultiMarkov object.
    pub fn build(mut self) -> MultiMarkov<T, R> {
        self.add_priors();
        MultiMarkov {
            markov_chain: self.markov_chain,
            known_states: self.known_states,
            order: self.order,
            rng: self.rng,
        }
    }

    /// Fills in missing state transitions with a given value so that any known state (except
    /// those only seen at the end of sequences) can transition to any other known state.
    /// Should be called after training is complete, because only then do we know the full set of
    /// known states, and which transitions are unobserved.
    fn add_priors(&mut self) {
        match self.prior {
            Some(p) => {
                for v in self.markov_chain.values_mut() {
                    for a in self.known_states.iter() {
                        v.entry(a.clone()).or_insert(p);
                    }
                }
            }
            None => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    fn char_data() -> Vec<Vec<char>> {
        vec![
            vec!['a'], // can't be used, but should be skipped over rather than causing error to propagate
            vec!['a', 'c', 'e'],
            vec!['f', 'o', 'o', 'b', 'a', 'r'],
            vec!['b', 'a', 'z'],
        ]
    }

    fn string_data() -> Vec<Vec<String>> {
        vec![
            vec![String::from("a")], // can't be used, but should be skipped over rather than causing error to propagate
            vec![String::from("a"), String::from("c"), String::from("e")],
            vec![
                String::from("f"),
                String::from("o"),
                String::from("o"),
                String::from("b"),
                String::from("a"),
                String::from("r"),
            ],
            vec![String::from("b"), String::from("a"), String::from("z")],
        ]
    }

    #[test]
    fn test_can_train_char_sequences() {
        let mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .train(char_data().into_iter());
    }

    #[test]
    fn test_can_train_string_sequences() {
        let mm = MultiMarkov::<String, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .train(string_data().into_iter());
    }

    #[test]
    fn sequences_in_training_show_up_in_model() {
        let mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .train(char_data().into_iter());
        // 'e' comes after 'c' (end of 2nd sequence trained properly)
        assert!(mm.markov_chain.get(&*vec!['c']).unwrap().contains_key(&'e'));
        // 'a' -> 'c' (beginning of 2nd sequence trained properly)
        assert!(mm.markov_chain.get(&*vec!['a']).unwrap().contains_key(&'c'));
        // a second-order sequence: ['a','c'] -> 'e'
        assert!(mm
            .markov_chain
            .get(&*vec!['a', 'c'])
            .unwrap()
            .contains_key(&'e'));
        // 'b' -> 'a' observed twice
        assert_eq!(
            *mm.markov_chain.get(&*vec!['b']).unwrap().get(&'a').unwrap(),
            2.0
        );
        // 'z' is in the alphabet of known states, but has no transitions because it was only seen at the end of a sequence
        assert!(mm.known_states.contains(&'z'));
        assert!(!mm.markov_chain.contains_key(&*vec!['z']));
        // we haven't added priors yet, so there should be no transition from 'a' -> 'b' available
        assert!(!mm.markov_chain.get(&*vec!['a']).unwrap().contains_key(&'b'));
    }

    #[test]
    fn can_set_priors_and_they_work() {
        let mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .train(char_data().into_iter())
            .with_prior(0.015)
            .build();
        // prior should be set for a non-observed transition such as 'a' -> 'b'
        assert!(mm.markov_chain.get(&*vec!['a']).unwrap().contains_key(&'b'));
        assert_eq!(
            *mm.markov_chain.get(&*vec!['a']).unwrap().get(&'b').unwrap(),
            0.015
        );
    }

    #[test]
    fn make_sure_it_works_with_strings_too() {
        let mm = MultiMarkov::<String, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .train(string_data().into_iter())
            .with_prior(0.011)
            .build();
        // prior should be set for a non-observed transition such as 'a' -> 'b'
        assert!(mm
            .markov_chain
            .get(&*vec![String::from("a")])
            .unwrap()
            .contains_key(&String::from("b")));
        assert_eq!(
            *mm.markov_chain
                .get(&*vec![String::from("a")])
                .unwrap()
                .get(&String::from("b"))
                .unwrap(),
            0.011
        );
    }

    #[test]
    fn can_specify_no_priors_and_build() {
        let mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(2)
            .train(char_data().into_iter())
            .without_prior()
            .build();
        // a non-observed transition such as 'a' -> 'b' should have no entry in the model
        assert!(!mm.markov_chain.get(&*vec!['a']).unwrap().contains_key(&'b'));
    }

    #[test]
    #[should_panic(expected = "Order must be an integer greater than zero.")]
    fn order_cannot_be_zero_or_negative() {
        let mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
            .with_order(0)
            .train(char_data().into_iter());
    }

    #[test]
    fn test_rng_clone() {
        use rand::{rngs::SmallRng, Rng, SeedableRng};
        let mut mm1 = MultiMarkov::<char, SmallRng>::builder(SmallRng::seed_from_u64(1234))
            .train(char_data().into_iter())
            .without_prior()
            .build();
        let mut mm2 = MultiMarkov::<char, SmallRng>::builder(SmallRng::seed_from_u64(1234))
            .train(char_data().into_iter())
            .without_prior()
            .build();
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
        assert_eq!(mm1.random_next(&vec!['a']), mm2.random_next(&vec!['a']));
    }
}
