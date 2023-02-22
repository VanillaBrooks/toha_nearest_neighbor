use kd_tree::KdTree;

use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;

use rayon::prelude::*;

pub fn kd_tree(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let kdtree = assemble_tree(line_points);

    let mut out = Array2::zeros(points_to_match.dim());

    let point_iter = points_to_match.axis_iter(Axis(0)).map(|point| {
        let point_x = point[[0]];
        let point_y = point[[1]];

        let item = kdtree.nearest(&[point_x, point_y]).unwrap();

        item.item
    });

    for (row, point) in point_iter.enumerate() {
        out[[row, 0]] = point[0];
        out[[row, 1]] = point[1];
    }

    out
}

pub fn kd_tree_par(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let kdtree = assemble_tree(line_points);

    let mut out = Array2::zeros(points_to_match.dim());

    let point_iter: Vec<[f64; 2]> = points_to_match
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|point| {
            let point_x = point[[0]];
            let point_y = point[[1]];

            let item = kdtree.nearest(&[point_x, point_y]).unwrap();

            item.item.clone()
        })
        // this allocation is not ideal here, but it seems to be unavoidable
        .collect();

    for (row, point) in point_iter.into_iter().enumerate() {
        out[[row, 0]] = point[0];
        out[[row, 1]] = point[1];
    }

    out
}

fn assemble_tree(line_points: ArrayView2<'_, f64>) -> KdTree<[f64; 2]> {
    let line_points: Vec<_> = line_points
        .axis_iter(Axis(0))
        .map(|point| {
            let point_x = point[[0]];
            let point_y = point[[1]];

            [point_x, point_y]
        })
        .collect();

    KdTree::build_by_ordered_float(line_points)
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn parallel_serial_same() {
        let lines = ndarray::Array2::random((10000, 2), Uniform::new(0.0, 10.0));
        let points = ndarray::Array2::random((3000, 2), Uniform::new(0.0, 10.0));

        let out_brute = kd_tree(lines.view(), points.view());
        let out_par = kd_tree_par(lines.view(), points.view());

        assert_eq!(out_brute, out_par);
    }
}
