use ndarray::ArrayView2;
use ndarray::Axis;

use rayon::prelude::*;

use super::FromShapeIter;
use super::IndexAndDistance;
use super::LocationAndDistance;
use super::SingleIndexDistance;
use super::SinglePointDistance;

pub fn brute_force_location<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> LocationAndDistance {
    brute_force::<DIM, SinglePointDistance<DIM>, LocationAndDistance>(line_points, points_to_match)
}

pub fn brute_force_index<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> IndexAndDistance {
    brute_force::<DIM, SingleIndexDistance, IndexAndDistance>(line_points, points_to_match)
}

pub fn brute_force_location_par<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> LocationAndDistance {
    brute_force_par::<DIM, SinglePointDistance<DIM>, LocationAndDistance>(
        line_points,
        points_to_match,
    )
}

pub fn brute_force_index_par<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> IndexAndDistance {
    brute_force_par::<DIM, SingleIndexDistance, IndexAndDistance>(line_points, points_to_match)
}

trait Distance {
    fn distance(&self) -> f64;
}

impl<const DIM: usize> Distance for super::SinglePointDistance<DIM> {
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
fn brute_force<'a, 'b, const DIM: usize, SINGLE, ALL>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> ALL
where
    SINGLE: From<(SingleIndexDistance, ArrayView2<'a, f64>)>,
    ALL: FromShapeIter<DIM, SINGLE>,
{
    let points_iter = points_to_match.axis_iter(Axis(0)).map(|point| {
        let array_point = super::copy_to_array::<DIM>(point);

        let min_distance = min_distance_to_point(line_points, array_point);
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
fn brute_force_par<'a, 'b, const DIM: usize, SINGLE, ALL>(
    line_points: ArrayView2<'a, f64>,
    points_to_match: ArrayView2<'b, f64>,
) -> ALL
where
    SINGLE: From<(SingleIndexDistance, ArrayView2<'a, f64>)> + Send,
    ALL: FromShapeIter<DIM, SINGLE>,
{
    let points_vec: Vec<_> = points_to_match
        .axis_iter(Axis(0))
        .into_iter()
        .into_par_iter()
        .map(|point| {
            assert!(point.len() == DIM);
            let array_point = super::copy_to_array::<DIM>(point);

            let min_distance = min_distance_to_point(line_points, array_point);
            SINGLE::from((min_distance, line_points))
        })
        .collect();

    ALL::from_shape_iter(points_vec, points_to_match.dim())
}

#[doc(hidden)]
pub fn min_distance_to_point<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    point: [f64; DIM],
) -> SingleIndexDistance {
    assert!(line_points.dim().1 == DIM);

    line_points
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(index, point_row)| {
            let line_point = super::copy_to_array::<DIM>(point_row);

            let mut distance = 0.0;

            for i in 0..DIM {
                distance += (point[i] - line_point[i]).powi(2);
            }

            SingleIndexDistance { distance, index }
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
        // the right float is NAN and the left float is fine (identical to `else` case)
        else {
            left
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let out_brute = brute_force_location::<2>(lines.view(), points.view());
        let out_par = brute_force_location_par::<2>(lines.view(), points.view());

        assert_eq!(out_brute, out_par);
    }
}
