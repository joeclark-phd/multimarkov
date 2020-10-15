use multimarkov::MultiMarkovModel;

fn main() {
    let mut model = MultiMarkovModel::<char>::new();

    let input_vec = vec![
        vec!['a','c','e'],
        vec!['f','o','o','b','a','r'],
        vec!['b','a','z'],
    ];
    model.add_sequences(input_vec).unwrap_or_else(|err| println!("Problem training model: {}", err));
    model.add_priors(MultiMarkovModel::<char>::DEFAULT_PRIOR);
    println!("frequencies: {:?}",model.frequencies);
    println!("known states: {:?}",model.known_states);
    println!("random next for 'abba': {:?}", model.random_next(&vec!['a','b','b','a']) );

}


