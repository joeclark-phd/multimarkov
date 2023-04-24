# multimarkov

This is a generic tool for training and using multi-order Markov chains for procedural generation applications such as generating randomized but authentic-sounding names for characters and locations.

It is a new implementation of [my markovmodels project (in Java)](https://github.com/joeclark-phd/markovmodels) but now in Rust.

## Release notes:

0.2.1: method `random_next()` no longer borrows self as mutable

0.2.0: You can now train the model more than once, for cumulative training using two data sets.  The new "build" method now returns a MarkovModel instead of a Result<MarkovModel>, so the API has changed a bit.

0.1.0: First release.