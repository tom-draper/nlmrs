use std::time::Instant;

fn ensure_examples_dir() {
    std::fs::create_dir_all("examples").expect("failed to create examples/ directory");
}

#[test]
fn test_write_to_csv() {
    ensure_examples_dir();
    let start = Instant::now();
    let arr = nlmrs::midpoint_displacement(1000, 1000, 0.6, None);
    let elapsed = start.elapsed();
    println!("elapsed: {:?}", elapsed);
    nlmrs::export::write_to_csv(&arr, "examples/example.csv").unwrap();
}

#[test]
fn test_write_to_json() {
    ensure_examples_dir();
    let start = Instant::now();
    let kernel = vec![vec![1., 0., 1.], vec![0., 0., 0.], vec![1., 0., 1.]];
    let arr = nlmrs::hill_grow(100, 100, 20000, true, Some(kernel), false, None);
    let elapsed = start.elapsed();
    println!("elapsed: {:?}", elapsed);
    nlmrs::export::write_to_json(&arr, "examples/example.json").unwrap();
}

#[test]
fn test_write_to_png() {
    ensure_examples_dir();
    let arr = nlmrs::midpoint_displacement(200, 200, 0.8, Some(42));
    nlmrs::export::write_to_png(&arr, "examples/example.png").unwrap();
}

#[test]
fn test_write_to_png_grayscale() {
    ensure_examples_dir();
    let arr = nlmrs::fbm_noise(200, 200, 4.0, 6, 0.5, 2.0, Some(42));
    nlmrs::export::write_to_png_grayscale(&arr, "examples/example_grayscale.png").unwrap();
}
