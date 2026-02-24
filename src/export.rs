use std::fs::File;
use std::io::{BufWriter, Result, Write};

use csv::Writer;

use crate::grid::Grid;

pub fn display(grid: &Grid) {
    for i in 0..grid.rows {
        println!("{:?}", &grid[i]);
    }
}

pub fn write_to_csv(grid: &Grid, path: &str) -> Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for i in 0..grid.rows {
        let _ = wtr.serialize(&grid[i]);
    }
    Ok(())
}

pub fn write_to_json(grid: &Grid, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    // Serialise as a 2D array for compatibility
    let rows: Vec<&[f64]> = (0..grid.rows).map(|i| &grid[i]).collect();
    serde_json::to_writer(&mut writer, &rows)?;
    writer.flush()?;
    Ok(())
}

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
    use image::{Rgb, RgbImage};
    let mut img = RgbImage::new(grid.cols as u32, grid.rows as u32);
    for i in 0..grid.rows {
        for j in 0..grid.cols {
            img.put_pixel(j as u32, i as u32, Rgb(terrain_color(grid[i][j])));
        }
    }
    img.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

/// Writes the grid as a grayscale PNG.
pub fn write_to_png_grayscale(grid: &Grid, path: &str) -> Result<()> {
    use image::{GrayImage, Luma};
    let mut img = GrayImage::new(grid.cols as u32, grid.rows as u32);
    for i in 0..grid.rows {
        for j in 0..grid.cols {
            let v = (grid[i][j].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(j as u32, i as u32, Luma([v]));
        }
    }
    img.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}
