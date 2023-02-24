mod brute_force;
mod pybind;
mod tree;

pub use brute_force::{brute_force, brute_force_par};
pub use tree::{kd_tree_index, kd_tree_index_par, kd_tree_location, kd_tree_location_par};

use ndarray::Array1;
use ndarray::Array2;

pub trait FromShapeIter<A> {
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = A>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexAndDistance {
    pub index: Array1<usize>,
    pub distance: Array1<f64>,
}

impl<'a> FromShapeIter<SingleIndexDistance> for IndexAndDistance {
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

impl<'a> FromShapeIter<SinglePointDistanceRef<'a>> for LocationAndDistance {
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = SinglePointDistanceRef<'a>>,
    {
        let iter = iter.into_iter();
        let mut location = Array2::zeros(cloud_shape);
        let mut distance = Array1::zeros(cloud_shape.0);

        for (row, point) in iter.enumerate() {
            location[[row, 0]] = point.point[0];
            location[[row, 1]] = point.point[1];

            distance[[row]] = point.distance;
        }

        LocationAndDistance { location, distance }
    }
}

impl FromShapeIter<SinglePointDistance> for LocationAndDistance {
    fn from_shape_iter<T>(iter: T, cloud_shape: (usize, usize)) -> Self
    where
        T: IntoIterator<Item = SinglePointDistance>,
    {
        let iter = iter.into_iter();
        let mut location = Array2::zeros(cloud_shape);
        let mut distance = Array1::zeros(cloud_shape.0);

        for (row, point) in iter.enumerate() {
            location[[row, 0]] = point.point[0];
            location[[row, 1]] = point.point[1];

            distance[[row]] = point.distance;
        }

        LocationAndDistance { location, distance }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct SinglePointDistanceRef<'a> {
    point: &'a [f64; 2],
    distance: f64,
}

#[derive(Debug, PartialEq, Clone)]
pub struct SinglePointDistance {
    point: [f64; 2],
    distance: f64,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct SingleIndexPointDistance {
    point: [f64; 2],
    index: usize,
    distance: f64,
}

#[derive(Debug, PartialEq, Clone)]
pub struct SingleIndexDistance {
    index: usize,
    distance: f64,
}

impl From<SingleIndexPointDistance> for SingleIndexDistance {
    fn from(x: SingleIndexPointDistance) -> Self {
        let SingleIndexPointDistance {
            index,
            distance,
            point: _,
        } = x;
        Self { index, distance }
    }
}

impl From<SingleIndexPointDistance> for SinglePointDistance {
    fn from(x: SingleIndexPointDistance) -> Self {
        let SingleIndexPointDistance {
            index: _,
            distance,
            point,
        } = x;
        Self { point, distance }
    }
}
