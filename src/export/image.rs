use std::io::Result;

use crate::grid::Grid;

/// Maps a normalised value [0, 1] to an RGB terrain colour.
///
/// Colours roughly approximate the matplotlib 'terrain' colormap:
/// deep water → shallow water → sand → grass → forest → rock → snow.
fn terrain_color(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let stops: &[(f64, [u8; 3])] = &[
        (0.00, [51, 51, 153]),   // deep water
        (0.20, [102, 153, 255]), // shallow water
        (0.25, [230, 220, 160]), // beach / sand
        (0.40, [80, 160, 50]),   // grass
        (0.65, [60, 110, 40]),   // forest
        (0.75, [150, 140, 110]), // rock
        (0.90, [160, 150, 140]), // high rock
        (1.00, [255, 255, 255]), // snow
    ];

    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            return [
                (c0[0] as f64 + f * (c1[0] as f64 - c0[0] as f64)).round() as u8,
                (c0[1] as f64 + f * (c1[1] as f64 - c0[1] as f64)).round() as u8,
                (c0[2] as f64 + f * (c1[2] as f64 - c0[2] as f64)).round() as u8,
            ];
        }
    }
    [255, 255, 255]
}

/// Writes the grid as a PNG using a terrain colormap (water → sand → grass → rock → snow).
pub fn write_to_png(grid: &Grid, path: &str) -> Result<()> {
    use image::RgbImage;
    let buf: Vec<u8> = grid.data.iter().flat_map(|&v| terrain_color(v)).collect();
    let img = RgbImage::from_raw(grid.cols as u32, grid.rows as u32, buf)
        .expect("buffer size mismatch");
    img.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

/// Writes the grid as a grayscale PNG.
pub fn write_to_png_grayscale(grid: &Grid, path: &str) -> Result<()> {
    use image::GrayImage;
    let buf: Vec<u8> = grid
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let img = GrayImage::from_raw(grid.cols as u32, grid.rows as u32, buf)
        .expect("buffer size mismatch");
    img.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

/// Writes the grid as a 16-bit grayscale TIFF.
///
/// Each cell value [0, 1] is mapped to the full u16 range [0, 65535], preserving
/// far more precision than an 8-bit PNG.
pub fn write_to_tiff(grid: &Grid, path: &str) -> Result<()> {
    use image::{ImageBuffer, Luma};
    let buf: Vec<u16> = grid
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
        .collect();
    let img: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_raw(grid.cols as u32, grid.rows as u32, buf)
            .expect("buffer size mismatch");
    img.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}
