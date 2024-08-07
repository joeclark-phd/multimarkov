use multimarkov::MultiMarkov;
use rand::{rngs::SmallRng, SeedableRng};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {

    // initialize logging
    env_logger::builder().filter_level(log::LevelFilter::Debug).init();


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

    let mut mm = MultiMarkov::<char>::builder()
        .with_order(3)
        .with_prior(0.02)
        .with_rng(Box::new(SmallRng::seed_from_u64(1234)))
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
