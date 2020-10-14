use multimarkov::MarkovModel;

fn main() {
    let mut model = MarkovModel::new();

    //let input_string = "Hello, world!";
    //model.add_sequence(input_string).unwrap_or_else(|err| println!("Problem training model: {}", err));

    let input_vec = vec!["ace","foobar","baz"];
    model.add_sequences(input_vec).unwrap_or_else(|err| println!("Problem training model: {}", err));

    //println!("frequencies: {:?}",model.frequencies);
    //println!("alphabet: {:?}",model.alphabet);

    println!("random next for 'abba': {:?}", model.random_next(&vec!['a','b','b','a']) );
    println!("random next for 'abba': {:?}", model.random_next(&vec!['a','b','b','a']) );
    println!("random next for 'abba': {:?}", model.random_next(&vec!['a','b','b','a']) );
    println!("random next for 'abba': {:?}", model.random_next(&vec!['a','b','b','a']) );
}


