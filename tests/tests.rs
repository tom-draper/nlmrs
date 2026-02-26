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

#[test]
fn test_write_to_tiff() {
    ensure_examples_dir();
    let arr = nlmrs::midpoint_displacement(200, 200, 0.8, Some(42));
    nlmrs::export::write_to_tiff(&arr, "examples/example.tif").unwrap();
}

#[test]
fn test_write_to_ascii_grid() {
    ensure_examples_dir();
    let arr = nlmrs::midpoint_displacement(200, 200, 0.8, Some(42));
    nlmrs::export::write_to_ascii_grid(&arr, "examples/example.asc").unwrap();

    // Verify the header is well-formed
    let content = std::fs::read_to_string("examples/example.asc").unwrap();
    let mut lines = content.lines();
    assert!(lines.next().unwrap().starts_with("ncols"));
    assert!(lines.next().unwrap().starts_with("nrows"));
    assert!(lines.next().unwrap().starts_with("xllcorner"));
    assert!(lines.next().unwrap().starts_with("yllcorner"));
    assert!(lines.next().unwrap().starts_with("cellsize"));
    assert!(lines.next().unwrap().starts_with("NODATA_value"));
    // First data row has 200 space-separated values
    let data_row = lines.next().unwrap();
    assert_eq!(data_row.split_whitespace().count(), 200);
}

#[test]
fn test_csv_round_trip() {
    ensure_examples_dir();
    let original = nlmrs::midpoint_displacement(50, 80, 0.8, Some(42));
    nlmrs::export::write_to_csv(&original, "examples/roundtrip.csv").unwrap();
    let loaded = nlmrs::export::read_from_csv("examples/roundtrip.csv").unwrap();

    assert_eq!(loaded.rows, original.rows);
    assert_eq!(loaded.cols, original.cols);
    for (a, b) in original.data.iter().zip(loaded.data.iter()) {
        assert!((a - b).abs() < 1e-9, "value mismatch: {a} vs {b}");
    }
}

#[test]
fn test_ascii_grid_round_trip() {
    ensure_examples_dir();
    let original = nlmrs::fbm_noise(60, 70, 4.0, 6, 0.5, 2.0, Some(7));
    nlmrs::export::write_to_ascii_grid(&original, "examples/roundtrip.asc").unwrap();
    let loaded = nlmrs::export::read_from_ascii_grid("examples/roundtrip.asc").unwrap();

    assert_eq!(loaded.rows, original.rows);
    assert_eq!(loaded.cols, original.cols);
    // ASCII grid is written with 6 decimal places so tolerance is 1e-6.
    for (a, b) in original.data.iter().zip(loaded.data.iter()) {
        assert!((a - b).abs() < 1e-5, "value mismatch: {a} vs {b}");
    }
}
