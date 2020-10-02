use std::collections::HashMap;

fn main() {
    let mut letter_counts: HashMap<char,i32> = HashMap::new();

    let input_string = "Hello, world!";
    let char_vec: Vec<char> = input_string.to_lowercase().chars().collect();
    for c in char_vec {
        if let Some(x) = letter_counts.get_mut(&c) {
            *x = *x + 1;
        } else {
            letter_counts.insert(c,1);
        }
    }
    println!("{:?}",letter_counts);
}
