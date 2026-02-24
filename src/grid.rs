use std::ops::{Index, IndexMut};

/// A 2D grid backed by a flat `Vec<f64>` for cache-friendly storage.
///
/// Indexing with `grid[row][col]` works naturally via `Index<usize>`.
pub struct Grid {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Grid {
    /// Creates a new grid filled with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Grid {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a new grid filled with the given value.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Grid {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    /// Returns true if the grid has no cells.
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Iterates over all cell values in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }

    /// Mutably iterates over all cell values in row-major order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.data.iter_mut()
    }
}

/// `grid[row]` returns a row slice, enabling `grid[row][col]` indexing.
impl Index<usize> for Grid {
    type Output = [f64];

    fn index(&self, row: usize) -> &[f64] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

impl IndexMut<usize> for Grid {
    fn index_mut(&mut self, row: usize) -> &mut [f64] {
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }
}
