use multimarkov::MultiMarkovModel;
use std::fs::File;
use std::io::{BufReader, BufRead};

fn main() {
    let mut model = MultiMarkovModel::<char>::new();

    let file = File::open("resources/romans.txt").unwrap();
    let reader = BufReader::new(file);
    let lines = reader.lines().map(|l| l.unwrap().to_lowercase() ).map(|l| l.chars().collect::<Vec<_>>()).map(|mut v| { v.insert(0, '#'); v.push('#'); v });
    model.add_sequences(lines).unwrap_or_else(|err| println!("Problem training model: {}", err));
    model.add_priors(MultiMarkovModel::<char>::DEFAULT_PRIOR);

    for _i in 0..10 {
        // generate a roman-sounding name
        let mut name = vec!['#']; // the beginning-of-word and end-of-word character
        name.push(model.random_next(&name).unwrap());
        while !name.ends_with(&*vec!['#']) {
            name.push(model.random_next(&name).unwrap());
        }
        name.pop();
        name.remove(0);
        let stringname = name.iter().collect::<String>();
        println!("{}", stringname);
    }
}


