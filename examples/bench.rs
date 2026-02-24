use std::time::Instant;

macro_rules! time {
    ($label:expr, $expr:expr) => {{
        let start = Instant::now();
        let result = $expr;
        let elapsed = start.elapsed();
        println!("{:<40} {:>8.0}ms", $label, elapsed.as_millis());
        result
    }};
}

fn main() {
    println!("\n{:<40} {:>8}", "Algorithm (release build)", "Time");
    println!("{}", "-".repeat(52));

    time!("random 2000x2000",          nlmrs::random(2000, 2000, Some(42)));
    time!("random_element 1000x1000",  nlmrs::random_element(1000, 1000, 50000., Some(42)));
    time!("planar_gradient 2000x2000", nlmrs::planar_gradient(2000, 2000, Some(45.), Some(42)));
    time!("edge_gradient 2000x2000",   nlmrs::edge_gradient(2000, 2000, Some(45.), Some(42)));
    time!("wave_gradient 2000x2000",   nlmrs::wave_gradient(2000, 2000, 2.5, Some(45.), Some(42)));
    time!("distance_gradient 500x500", nlmrs::distance_gradient(500, 500, Some(42)));
    time!("midpoint_disp 1000x1000",   nlmrs::midpoint_displacement(1000, 1000, 1.0, Some(42)));
    time!("hill_grow 1000x1000 n=50k", nlmrs::hill_grow(1000, 1000, 50000, true, None, false, Some(42)));
    time!("perlin_noise 2000x2000",    nlmrs::perlin_noise(2000, 2000, 4.0, Some(42)));
    time!("fbm_noise 2000x2000 oct=8", nlmrs::fbm_noise(2000, 2000, 4.0, 8, 0.5, 2.0, Some(42)));

    println!();
}
