// based on https://www.youtube.com/watch?v=KDn_j48yoAo
fn main() {
    // variables are immutable by default, unless marked with `mut`
    let x = 3 * -5;    // integer
    let y = 2.1;       // float

    let boo = true; // boolean

    let arr = ['a', 'b', 'r']; // must all be same type
    println!("Hello, world! x = {}, y = {}, boo = {}", x, y, boo);
    //println!("{}", arr);

    let my_tuple = (3, 4.1, "Dan"); // rust hates camel case
    println!("{}", my_tuple.0);
    say_hello("Bob");
    add(11, -4);
}

//fn say_hello(name: str) { // complains "doesn't have a size known at compile-time" for the println below
fn say_hello(name: &str) { // makes it work (dynamically figuring out the size at runtime)
    println!("hello {}", name);
}

fn add(x: i8, y: i8) {
    println!("{} + {} = {}", x, y, x+y);

}