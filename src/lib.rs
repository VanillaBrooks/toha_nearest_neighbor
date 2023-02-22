mod brute_force;
mod pybind;
mod tree;

pub use brute_force::{brute_force, brute_force_par};
pub use tree::{kd_tree, kd_tree_par};

use ndarray::Array2;

fn arr2_from_iter<'a>(
    iter: impl Iterator<Item = &'a [f64; 2]>,
    shape: (usize, usize),
) -> Array2<f64> {
    let mut out = Array2::zeros(shape);

    for (row, point) in iter.enumerate() {
        out[[row, 0]] = point[0];
        out[[row, 1]] = point[1];
    }

    out
}

fn arr2_from_iter_owned(
    iter: impl Iterator<Item = [f64; 2]>,
    shape: (usize, usize),
) -> Array2<f64> {
    let mut out = Array2::zeros(shape);

    for (row, point) in iter.enumerate() {
        out[[row, 0]] = point[0];
        out[[row, 1]] = point[1];
    }

    out
}
