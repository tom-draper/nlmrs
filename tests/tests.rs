use std::time::Instant;

#[test]
fn test_write_to_csv() {
    let start = Instant::now();

    // let arr = nlmrs::random(100, 100);
    // let arr = nlmrs::random_element(100, 100, 50000.);
    // let arr = nlmrs::planar_gradient(100, 100, Some(60.));
    // let arr = nlmrs::edge_gradient(100, 100, Some(140.));
    // let arr = nlmrs::distance_gradient(100, 100);
    // let arr = nlmrs::wave_gradient(100, 100, 2.5, Some(90.));
    // let arr = nlmrs::midpoint_displacement(100, 100, 0.9);
    let kernel = vec![vec![1., 1., 1.], vec![1., 1., 1.], vec![1., 1., 1.]];
    let arr = nlmrs::hill_grow(100, 100, 20000,true, Some(kernel), Some(false));

    let elapsed = start.elapsed();
    println!("elapsed: {:?}", elapsed);
    nlmrs::export::write_to_csv(arr);
}
