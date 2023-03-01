#[doc(hidden)]
pub mod brute_force;
mod pybind;
mod tree;

pub use brute_force::{
    brute_force_index, brute_force_index_par, brute_force_location, brute_force_location_par,
};
pub use tree::{kd_tree_index, kd_tree_index_par, kd_tree_location, kd_tree_location_par};

use ndarray::Array1;
use ndarray::Array2;

pub trait FromShapeIter<const DIM: usize, A> {
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = A>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexAndDistance {
    pub index: Array1<usize>,
    pub distance: Array1<f64>,
}

impl<const DIM: usize> FromShapeIter<DIM, SingleIndexDistance> for IndexAndDistance {
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = SingleIndexDistance>,
    {
        let iter = iter.into_iter();
        let mut index = Array1::zeros(cloud_shape.0);
        let mut distance = Array1::zeros(cloud_shape.0);

        for (row, point) in iter.enumerate() {
            index[[row]] = point.index;
            distance[[row]] = point.distance;
        }

        IndexAndDistance { index, distance }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocationAndDistance {
    pub location: Array2<f64>,
    pub distance: Array1<f64>,
}

impl<'a, const DIM: usize> FromShapeIter<DIM, SinglePointDistanceRef<'a, DIM>>
    for LocationAndDistance
{
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = SinglePointDistanceRef<'a, DIM>>,
    {
        let iter = iter.into_iter();
        let mut location = Array2::zeros(cloud_shape);
        let mut distance = Array1::zeros(cloud_shape.0);

        for (row, point) in iter.enumerate() {
            for col in 0..DIM {
                location[[row, col]] = point.point[col];
            }

            distance[[row]] = point.distance;
        }

        LocationAndDistance { location, distance }
    }
}

impl<const DIM: usize> FromShapeIter<DIM, SinglePointDistance<DIM>> for LocationAndDistance {
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = SinglePointDistance<DIM>>,
    {
        let iter = iter.into_iter();
        let mut location = Array2::zeros(cloud_shape);
        let mut distance = Array1::zeros(cloud_shape.0);

        for (row, point) in iter.enumerate() {
            for col in 0..DIM {
                location[[row, col]] = point.point[col];
            }

            distance[[row]] = point.distance;
        }

        LocationAndDistance { location, distance }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct SinglePointDistanceRef<'a, const DIM: usize> {
    point: &'a [f64; DIM],
    distance: f64,
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct SinglePointDistance<const DIM: usize> {
    point: [f64; DIM],
    distance: f64,
}

impl<const DIM: usize> From<(SingleIndexDistance, ndarray::ArrayView2<'_, f64>)>
    for SinglePointDistance<DIM>
{
    fn from(x: (SingleIndexDistance, ndarray::ArrayView2<'_, f64>)) -> Self {
        let (point_distance, array) = x;

        let mut point: [f64; DIM] = [0.; DIM];

        for col in 0..DIM {
            // because these indexes from the compiler's perspective can be random,
            // its helpful to use unsafe indexing on the ndarray::Array
            point[col] = unsafe { *array.uget([point_distance.index, col]) }
        }

        SinglePointDistance {
            point,
            distance: point_distance.distance,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct SingleIndexPointDistance {
    point: [f64; 2],
    index: usize,
    distance: f64,
}

#[derive(Debug, PartialEq, Clone)]
#[doc(hidden)]
pub struct SingleIndexDistance {
    index: usize,
    distance: f64,
}

impl From<(SingleIndexDistance, ndarray::ArrayView2<'_, f64>)> for SingleIndexDistance {
    fn from(x: (SingleIndexDistance, ndarray::ArrayView2<'_, f64>)) -> Self {
        x.0
    }
}

#[inline]
fn copy_to_array<const DIM: usize>(arr: ndarray::ArrayView1<'_, f64>) -> [f64; DIM] {
    #[cfg(debug_assertions)]
    assert_eq!(arr.is_standard_layout(), true);

    let mut array_point = [0.; DIM];

    for i in 0..DIM {
        array_point[i] = arr[i]
    }

    array_point
}
