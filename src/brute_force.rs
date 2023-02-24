use ndarray::ArrayView2;
use ndarray::Axis;

use rayon::prelude::*;

use super::FromShapeIter;
use super::SingleIndexDistance;
use super::SinglePointDistance;
use super::LocationAndDistance;
use super::IndexAndDistance;

pub fn brute_force_location<'a, 'b>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> LocationAndDistance {
    brute_force::<SinglePointDistance, LocationAndDistance>(line_points, points_to_match)
}

pub fn brute_force_index<'a, 'b>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> IndexAndDistance{
    brute_force::<SingleIndexDistance, IndexAndDistance>(line_points, points_to_match)
}

pub fn brute_force_location_par<'a, 'b>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> LocationAndDistance {
    brute_force_par::<SinglePointDistance, LocationAndDistance>(line_points, points_to_match)
}

pub fn brute_force_index_par<'a, 'b>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> IndexAndDistance{
    brute_force_par::<SingleIndexDistance, IndexAndDistance>(line_points, points_to_match)
}

trait Distance {
    fn distance(&self) -> f64;
}

impl Distance for super::SinglePointDistance {
    fn distance(&self) -> f64 {
        self.distance
    }
}

impl Distance for super::SingleIndexDistance {
    fn distance(&self) -> f64 {
        self.distance
    }
}

/// Brute force the nearest neighbor problem with serial iteration
///
/// ## Parameters
///
/// `line_points` is the 2D array of datapoints that are candidates for the 2D array of points in
/// `points_to_match`. Essentially, every row of `points_to_match` contains two columns (x,y
/// location floats) that will be matched against all rows of `line_points` (in the same format)
/// to find the minimum distance.
///
/// `SINGLE` is the type description for a single datapoint (its index and distance, or its point
/// location and distance). `ALL` is the collection of all `SINGLE` points to an array format.
///
/// Usually `SINGLE` = [`SinglePointDistance`] (`ALL` = `[LocationAndDistance`]), or
/// `SINGLE` = [`SingleIndexDistance`] (`ALL` = `[IndexAndDistance]`).
fn brute_force<'a, 'b, SINGLE, ALL>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> ALL
where
    SINGLE: From<(SingleIndexDistance, ArrayView2<'a, f64>)>,
    ALL: FromShapeIter<SINGLE>,
{
    let points_iter = points_to_match.axis_iter(Axis(0)).map(|point| {
        let point_x = point[[0]];
        let point_y = point[[1]];

        let min_distance = min_distance_to_point(line_points, [point_x, point_y]);
        SINGLE::from((min_distance, line_points))

    });

    ALL::from_shape_iter(points_iter, points_to_match.dim())
}

/// Brute force the nearest neighbor problem with parallel iteration
///
/// ## Parameters
///
/// `line_points` is the 2D array of datapoints that are candidates for the 2D array of points in
/// `points_to_match`. Essentially, every row of `points_to_match` contains two columns (x,y
/// location floats) that will be matched against all rows of `line_points` (in the same format)
/// to find the minimum distance.
///
/// `SINGLE` is the type description for a single datapoint (its index and distance, or its point
/// location and distance). `ALL` is the collection of all `SINGLE` points to an array format.
///
/// Usually `SINGLE` = [`SinglePointDistance`] (`ALL` = `[LocationAndDistance`]), or
/// `SINGLE` = [`SingleIndexDistance`] (`ALL` = `[IndexAndDistance]`).
fn brute_force_par<'a, 'b, SINGLE, ALL>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> ALL
where
    SINGLE: From<(SingleIndexDistance, ArrayView2<'a, f64>)> + Send,
    ALL: FromShapeIter<SINGLE>,
{
    let points_vec: Vec<_> = points_to_match
        .axis_iter(Axis(0))
        .into_iter()
        .into_par_iter()
        .map(|point| {
            let point_x = point[[0]];
            let point_y = point[[1]];

            let min_distance = min_distance_to_point(line_points, [point_x, point_y]);
            SINGLE::from((min_distance, line_points))
        })
        .collect();

    ALL::from_shape_iter(points_vec, points_to_match.dim())
}

fn min_distance_to_point(
    line_points: ArrayView2<'_, f64>,
    point: [f64; 2],
) -> SingleIndexDistance {
    line_points
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(index, point_row)| {
            let line_x = point_row[[0]];
            let line_y = point_row[[1]];

            let line_point = [line_x, line_y];

            let distance = (point[0] - line_point[0]).powi(2) + (point[1] - line_point[1]).powi(2);

            SingleIndexDistance {
                distance,
                index,
            }
        })
        .reduce(minimize_float)
        .unwrap()
}

fn minimize_float<T: Distance>(left: T, right: T) -> T {
    let left_float: f64 = left.distance();
    let right_float: f64 = right.distance();

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
    use crate::LocationAndDistance;
    use crate::SinglePointDistance;
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn minimize_left() {
        let left = SinglePointDistance {
            distance: 0.,
            point: [0., 0.],
        };
        let right = SinglePointDistance {
            distance: 1.,
            point: [1., 1.],
        };

        let out = minimize_float(left, right);

        assert_eq!(out, left)
    }

    #[test]
    fn minimize_right() {
        let left = SinglePointDistance {
            distance: 1.,
            point: [0., 0.],
        };
        let right = SinglePointDistance {
            distance: 0.,
            point: [1., 1.],
        };

        let out = minimize_float(left, right);

        assert_eq!(out, right)
    }

    #[test]
    fn minimize_eq() {
        let left = SinglePointDistance {
            distance: 0.,
            point: [0., 0.],
        };
        let right = SinglePointDistance {
            distance: 0.,
            point: [1., 1.],
        };

        let out = minimize_float(left, right);

        assert_eq!(out, left)
    }

    #[test]
    fn minimize_left_nan() {
        let left = SinglePointDistance {
            distance: f64::NAN,
            point: [0., 0.],
        };
        let right = SinglePointDistance {
            distance: 0.,
            point: [1., 1.],
        };

        let out = minimize_float(left, right);

        assert_eq!(out, right)
    }

    #[test]
    fn minimize_right_nan() {
        let left = SinglePointDistance {
            distance: 20.,
            point: [0., 0.],
        };
        let right = SinglePointDistance {
            distance: f64::NAN,
            point: [0., 0.],
        };

        let out = minimize_float(left, right);

        assert_eq!(out, left)
    }

    #[test]
    fn nearest_neighbor_single() {
        let line_points = ndarray::arr2(&[[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]);

        let point = [1.1, 0.1];

        let out = min_distance_to_point(line_points.view(), point);
        let out = crate::SinglePointDistance::from((out, line_points.view()));

        assert_eq!(out.point, [1.0, 0.0]);
    }

    #[test]
    fn parallel_serial_same() {
        let lines = Array2::random((10000, 2), Uniform::new(0.0, 10.0));
        let points = Array2::random((3000, 2), Uniform::new(0.0, 10.0));

        let out_brute =
            brute_force::<SinglePointDistance, LocationAndDistance>(lines.view(), points.view());
        let out_par = brute_force_par::<SinglePointDistance, LocationAndDistance>(
            lines.view(),
            points.view(),
        );

        assert_eq!(out_brute, out_par);
    }
}
