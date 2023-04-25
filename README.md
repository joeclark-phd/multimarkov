# multimarkov

This is a generic tool for training and using multi-order Markov chains for procedural generation applications such as generating randomized but authentic-sounding names for characters and locations.

It is a new implementation of [my markovmodels project (in Java)](https://github.com/joeclark-phd/markovmodels) but now in Rust.

## Markov chains

A Markov chain maps current states to possible future states, usually defined with probabilities ([c.f. wikipedia](https://en.wikipedia.org/wiki/Markov_chain)).  This is useful in procedural generation, for example to model which letters in a language most frequently follow a given letter, or to model which weather conditions most likely follow a given weather condition.  For each known state, a MarkovChain knows or can calculate the possible following states, so a client should be able to traverse the "chain" as many iterations as he likes (e.g. for a simulation).  In most cases these transitions will be weighted by a probability distribution or observed frequencies, so not all transitions from any given state will be equally likely.

## `MultiMarkov<T>`

`MultiMarkov` is the key struct in this crate.  It is inspired by the algorithm [described by JLund3 at RogueBasin](http://www.roguebasin.com/index.php/Names_from_a_high_order_Markov_Process_and_a_simplified_Katz_back-off_scheme).

This implementation offers multi-order models with a Katz back-off.  What that means is, if `order` is greater than 1, multiple models may match any given sequence. For example, if you provide the sequence `['R,'U','S','T']`, and `order` equals `3`, we will first check to see if we have a model for states that follow `['U','S','T']`. If none is found, for example because that sequence was never seen in training data, we check for a model for states that follow `['S','T']`. Failing that too, we fall back on the model of transitions from the state `'T'`, which is certain to exist if `'T'` was even once observed in the training data with a following state.

Another feature that may be desired in procedural generation applications is the option to inject some "true randomness" in the form of "prior" relative probabilities, i.e., small weights given to transitions *not* observed in training data. These can make up for the limitations of a training dataset and enable the generation of sequences not observed in training.  By default, we assign a probability of 0.005 to each such transition.

If multi-order models are not desired, use `.with_order(1)`.  If priors are not desired, use `.without_prior()`.

## Usage

Add `multimarkov` to your `Cargo.toml`.

### Building and training

To build a `MultiMarkov` instance, use the builder pattern.  `T` can be any type that implements `Eq + Hash + Clone`.  Here we are using `char`:

    let training_data = vec![
        vec!['f','o','o','b','a','r'],
        vec!['s','n','a','f','u'],
    ];

    let mm = MultiMarkov::<char>::builder()
        .with_order(2) // omit to use default of 3
        .with_prior(0.01) // omit to use default of 0.005, or call .without_prior() to disable priors
        .train(input_vec.into_iter())
        .build();

### Procedural generation

To get a random draw, call `random_next()` with an `&Vec<T>` representing the current or previous state(s). For example:

    let next_letter = mm.random_next(&Vec['a']);

will randomly draw a letter to follow `'a'`.  Based on the training data, that will probably be `'r'` or `'f'`, but because of the "priors", any known state has a small chance of being drawn.

The reason `random_next` requires a vector is that you may be using a multi-order model that needs to look back a few states in the sequence.  For example:

    let next_letter = mm.random_next(&Vec['s','n','a']);

is much more likely to draw `'f'` because it has trained a model for what comes after `'n','a'` which it prefers to use rather than its model of what comes after `'a'`.



## Release notes:

0.3.0: Mostly rewritten; now `T` can be any `Eq + Hash + Clone`, and doesn't need `Copy`, which means we can use sequences of strings.  I also introduced a real "builder" struct, MultiMarkovBuilder.

0.2.1: method `random_next()` no longer borrows self as mutable

0.2.0: You can now train the model more than once, for cumulative training using two data sets.  The new "build" method now returns a MarkovModel instead of a Result<MarkovModel>, so the API has changed a bit.

0.1.0: First release.