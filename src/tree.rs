use kd_tree::KdTree;

use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;

use rayon::prelude::*;

use super::FromShapeIter;
use super::LocationAndDistance;
use super::SinglePointDistance;
use super::SinglePointDistanceRef;

pub fn kd_tree(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> LocationAndDistance {
    let kdtree = assemble_tree(line_points);

    let point_iter = points_to_match.axis_iter(Axis(0)).map(|point| {
        let point_x = point[[0]];
        let point_y = point[[1]];

        let item = kdtree.nearest(&[point_x, point_y]).unwrap();

        SinglePointDistanceRef { point: item.item, distance: item.squared_distance }
    });


    LocationAndDistance::from_shape_iter(point_iter, points_to_match.dim())
}

pub fn kd_tree_par(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> LocationAndDistance {
    let kdtree = assemble_tree(line_points);

    let points_vec: Vec<_> = points_to_match
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|point| {
            let point_x = point[[0]];
            let point_y = point[[1]];

            let item = kdtree.nearest(&[point_x, point_y]).unwrap();

            SinglePointDistance{ point: *item.item, distance: item.squared_distance }
        })
        // this allocation is not ideal here, but it seems to be unavoidable
        .collect();

    LocationAndDistance::from_shape_iter(points_vec, points_to_match.dim())
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

        let kd_brute = kd_tree(lines.view(), points.view());
        let kd_par = kd_tree_par(lines.view(), points.view());

        assert_eq!(kd_brute, kd_par);
    }

    #[test]
    fn kdtree_brute_force_same() {
        let lines = ndarray::Array2::random((100, 2), Uniform::new(0.0, 10.0));
        let points = ndarray::Array2::random((100, 2), Uniform::new(0.0, 10.0));

        let out_kd = kd_tree(lines.view(), points.view());
        let out_brute = crate::brute_force::<SinglePointDistance, _>(lines.view(), points.view());

        assert_eq!(out_kd, out_brute);
    }
}
