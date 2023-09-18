mod builder;

use multimarkov::MultiMarkov;
use rand::{rngs::ThreadRng, thread_rng};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let file = File::open("resources/romans.txt").unwrap();
    let reader = BufReader::new(file);
    let lines = reader
        .lines()
        .map(|l| l.unwrap().to_lowercase())
        .map(|l| l.chars().collect::<Vec<_>>())
        .map(|mut v| {
            v.insert(0, '#');
            v.push('#');
            v
        });

    let mut mm = MultiMarkov::<char, ThreadRng>::builder(thread_rng())
        .with_order(3)
        .with_prior(0.02)
        .train(lines)
        .build();

    for _i in 0..10 {
        // generate a roman-sounding name
        let mut name = vec!['#']; // the beginning-of-word and end-of-word character
        name.push(mm.random_next(&name).unwrap());
        while !name.ends_with(&*vec!['#']) {
            name.push(mm.random_next(&name).unwrap());
        }
        name.pop();
        name.remove(0);
        let stringname = name.iter().collect::<String>();
        println!("{}", stringname);
    }
}
