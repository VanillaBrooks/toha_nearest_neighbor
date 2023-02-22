use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;

use rayon::prelude::*;

pub fn brute_force<'a, 'b>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> Vec<usize> {
    points_to_match
        .axis_iter(Axis(0))
        .map(|point| {
            let point_x = point[[0]];
            let point_y = point[[1]];

            min_distance_to_point(line_points, point_x, point_y)
        })
        .collect()
}

pub fn brute_force_par<'a, 'b>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> Vec<usize> {
    points_to_match
        .axis_iter(Axis(0))
        .into_iter()
        .into_par_iter()
        .map(|point| {
            let point_x = point[[0]];
            let point_y = point[[1]];

            min_distance_to_point(line_points, point_x, point_y)
        })
        .collect()
}

fn min_distance_to_point(line_points: ArrayView2<'_, f64>, point_x: f64, point_y: f64) -> usize {
    line_points
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(idx, point_row)| {
            let line_x = point_row[[0]];
            let line_y = point_row[[1]];

            let distance = (point_x - line_x).powi(2) + (point_y - line_y).powi(2);
            (idx, distance)
        })
        .reduce(minimize_float)
        .map(|(idx, _distance)| idx)
        .unwrap()
}

fn minimize_float(left: (usize, f64), right: (usize, f64)) -> (usize, f64) {
    let left_float: f64 = left.1;
    let right_float: f64 = right.1;

    if left_float < right_float {
        left
    } else if right_float < left_float {
        right
    } else {
        // the left float is NAN and the right float is fine
        if left_float.is_nan() && !right_float.is_nan() {
            right
        }
        // the right float is NAN and the left float is fine
        else if right_float.is_nan() && !left_float.is_nan() {
            left
        }
        // both are NAN, just return the left one
        else {
            left
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn minimize_left() {
        let left = (0, 0.);
        let right = (1, 1.);

        let out = minimize_float(left, right);

        assert_eq!(out, left)
    }

    #[test]
    fn minimize_right() {
        let left = (0, 1.);
        let right = (1, 0.);

        let out = minimize_float(left, right);

        assert_eq!(out, right)
    }

    #[test]
    fn minimize_eq() {
        let left = (0, 0.);
        let right = (1, 0.);

        let out = minimize_float(left, right);

        assert_eq!(out, left)
    }

    #[test]
    fn minimize_left_nan() {
        let left = (0, f64::NAN);
        let right = (1, 0.);

        let out = minimize_float(left, right);

        assert_eq!(out, right)
    }

    #[test]
    fn minimize_right_nan() {
        let left = (0, 20.);
        let right = (1, f64::NAN);

        let out = minimize_float(left, right);

        assert_eq!(out, left)
    }

    #[test]
    fn nearest_neighbor_single() {
        let line_points = ndarray::arr2(&[[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]);

        let point_x = 1.1;
        let point_y = 0.1;

        let out = min_distance_to_point(line_points.view(), point_x, point_y);

        assert_eq!(out, 1);
    }

    #[test]
    fn parallel_serial_same() {
        let lines = ndarray::Array2::random((10000, 2), Uniform::new(0.0, 10.0));
        let points = ndarray::Array2::random((3000, 2), Uniform::new(0.0, 10.0));

        let out_brute = brute_force(lines.view(), points.view());
        let out_par = brute_force_par(lines.view(), points.view());

        assert_eq!(out_brute, out_par);
    }
}
