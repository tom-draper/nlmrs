use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Result, Write};

use csv::Writer;

use crate::grid::Grid;

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

/// Writes the grid as an ESRI ASCII Grid (`.asc`).
///
/// The format is widely supported by GIS and ecology software (R `terra`/`raster`,
/// QGIS, ArcGIS). No spatial reference is set — `xllcorner`, `yllcorner` default
/// to 0.0 and `cellsize` to 1.0.
pub fn write_to_ascii_grid(grid: &Grid, path: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "ncols         {}", grid.cols)?;
    writeln!(w, "nrows         {}", grid.rows)?;
    writeln!(w, "xllcorner     0.0")?;
    writeln!(w, "yllcorner     0.0")?;
    writeln!(w, "cellsize      1.0")?;
    writeln!(w, "NODATA_value  -9999")?;
    for i in 0..grid.rows {
        let row = &grid[i];
        for (j, &v) in row.iter().enumerate() {
            if j > 0 {
                w.write_all(b" ")?;
            }
            write!(w, "{:.6}", v)?;
        }
        w.write_all(b"\n")?;
    }
    w.flush()
}

// ── Import ────────────────────────────────────────────────────────────────────

/// Reads a grid from a CSV file written by [`write_to_csv`].
///
/// The file must contain rows of comma-separated `f64` values with no header row.
/// The grid dimensions are inferred from the number of rows and columns in the file.
pub fn read_from_csv(path: &str) -> Result<Grid> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let mut data: Vec<f64> = Vec::new();
    let mut cols = 0usize;
    let mut rows = 0usize;

    for result in rdr.records() {
        let record = result
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        if rows == 0 {
            cols = record.len();
        } else if record.len() != cols {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("row {} has {} columns, expected {}", rows + 1, record.len(), cols),
            ));
        }
        for field in record.iter() {
            data.push(field.trim().parse::<f64>().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
            })?);
        }
        rows += 1;
    }

    Ok(Grid { data, rows, cols })
}

/// Parse one "KEY  VALUE" header line from an ESRI ASCII Grid; returns the value string.
fn parse_asc_header(line: &str, expected_key: &str) -> Result<String> {
    let mut parts = line.split_whitespace();
    let key = parts.next().unwrap_or("");
    if !key.eq_ignore_ascii_case(expected_key) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected header '{expected_key}', found '{key}'"),
        ));
    }
    parts.next().map(|s| s.to_string()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("missing value for '{expected_key}'"),
        )
    })
}

/// Reads a grid from an ESRI ASCII Grid (`.asc`) file written by [`write_to_ascii_grid`].
///
/// Spatial metadata (xllcorner, yllcorner, cellsize, NODATA_value) is parsed and
/// discarded — only the grid dimensions and cell values are returned.
pub fn read_from_ascii_grid(path: &str) -> Result<Grid> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    let mut next_line = |key: &str| -> Result<String> {
        let raw = lines
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated header"))??;
        parse_asc_header(&raw, key)
    };

    let cols = next_line("ncols")?.parse::<usize>().map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })?;
    let rows = next_line("nrows")?.parse::<usize>().map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })?;
    next_line("xllcorner")?;
    next_line("yllcorner")?;
    next_line("cellsize")?;
    next_line("NODATA_value")?;

    let mut data = Vec::with_capacity(rows * cols);
    for line in lines {
        for tok in line?.split_whitespace() {
            data.push(tok.parse::<f64>().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
            })?);
        }
    }

    if data.len() != rows * cols {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected {} values, found {}", rows * cols, data.len()),
        ));
    }

    Ok(Grid { data, rows, cols })
}
