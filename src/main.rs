use multimarkov::MarkovModel;

fn main() {
    let mut model = MarkovModel::new();

    let input_string = "Hello, world!";
    model.add_sequence(input_string);

    println!("frequencies: {:?}",model.frequencies);
    println!("alphabet: {:?}",model.alphabet);
}


