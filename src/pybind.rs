use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (line_points, point_cloud, parallel = false))]
/// For every point in ``point_cloud``, find it's nearest neighbor in ``line_points`` using a
/// brute-force algorithm. Returns a tuple of the closest ``line_points`` locations and
/// their associated distances.
///
/// :param np.ndarray line_points: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_LINE_POINTS, 2)``
/// :param np.ndarray point_cloud: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_POINT_CLOUD_POINTS, 2)``
/// :param bool parallel: enable parallel processing for the dataset. If you have more than 2,000 line points and 2,000 point cloud points this may be useful.
///
/// this function returns  a ``(NUM_POINT_CLOUD_POINTS, 2)`` shaped array of the points where
/// each ``i`` th row of the returned array is a row of ``line_points`` that is closest to the
/// ``i`` th row of ``point_cloud``
///
/// Example:
///
/// .. code-block::
///     
///     import toha_nearest_neighbor
///     import numpy as np
///
///     line_points = np.array(
///         [
///             [0.0, 0.0],
///             [1.0, 1.0],
///             [2.0, 2.0],
///         ]
///     )
///
///     point_cloud = np.array(
///         [
///             [0.1, -0.1], #closest to the 0-th index of line_points rows
///             [2.2, 3.0], # closest to the 2-nd index of line_points rows
///         ]
///     )
///
///     closest_line_points, distances = toha_nearest_neighbor.brute_force(line_points, point_cloud)
///
///     # [[0. 0.]
///     #  [2. 2.]]
///     print(closest_line_points)
///     # [0.02 1.04]
///     print(distances)
///
fn brute_force_location<'a>(
    py: Python<'a>,
    line_points: PyReadonlyArray2<'_, f64>,
    point_cloud: PyReadonlyArray2<'_, f64>,
    parallel: bool,
) -> (&'a PyArray2<f64>, &'a PyArray1<f64>) {
    //let location_and_distance = if parallel {
    //    super::brute_force_location_par::<2>(line_points.as_array(), point_cloud.as_array())
    //} else {
    //    super::brute_force_location::<2>(line_points.as_array(), point_cloud.as_array())
    //};

    let location_and_distance = if parallel {
        macros::dimension_expansion!(
            super::brute_force_location_par(line_points.as_array(), point_cloud.as_array()),
            15
        )
    } else {
        macros::dimension_expansion!(
            super::brute_force_location(line_points.as_array(), point_cloud.as_array()),
            15
        )
    };

    let loc = PyArray2::from_owned_array(py, location_and_distance.location);
    let distance = PyArray1::from_owned_array(py, location_and_distance.distance);

    (loc, distance)
}

#[pyfunction]
#[pyo3(signature = (line_points, point_cloud, parallel = false))]
/// For every point in ``point_cloud``, find it's nearest neighbor in ``line_points`` using a
/// brute-force algorithm. Returns a tuple of indicies of the closest ``line_points`` rows and
/// their associated distances.
///
/// :param np.ndarray line_points: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_LINE_POINTS, 2)``
/// :param np.ndarray point_cloud: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_POINT_CLOUD_POINTS, 2)``
/// :param bool parallel: enable parallel processing for the dataset. If you have more than 2,000 line points and 2,000 point cloud points this may be useful.
///
/// Example:
///
/// .. code-block::
///     
///     import toha_nearest_neighbor
///     import numpy as np
///
///     line_points = np.array(
///         [
///             [0.0, 0.0],
///             [1.0, 1.0],
///             [2.0, 2.0],
///         ]
///     )
///
///     point_cloud = np.array(
///         [
///             [0.1, -0.1], #closest to the 0-th index of line_points rows
///             [2.2, 3.0], # closest to the 2-nd index of line_points rows
///         ]
///     )
///
///     indicies, distances = toha_nearest_neighbor.brute_force_index(line_points, point_cloud)
///     # [0 2]
///     print(indicies)
///     # [0.02 1.04]
///     print(distances)
///
fn brute_force_index<'a>(
    py: Python<'a>,
    line_points: PyReadonlyArray2<'_, f64>,
    point_cloud: PyReadonlyArray2<'_, f64>,
    parallel: bool,
) -> (&'a PyArray1<usize>, &'a PyArray1<f64>) {
    let index_and_distance = if parallel {
        macros::dimension_expansion!(
            super::brute_force_index_par(line_points.as_array(), point_cloud.as_array()),
            15
        )
    } else {
        macros::dimension_expansion!(
            super::brute_force_index(line_points.as_array(), point_cloud.as_array()),
            15
        )
    };

    let index = PyArray1::from_owned_array(py, index_and_distance.index);
    let distance = PyArray1::from_owned_array(py, index_and_distance.distance);

    (index, distance)
}

#[pyfunction]
#[pyo3(signature = (line_points, point_cloud, parallel = false))]
/// For every point in ``point_cloud``, find it's nearest neighbor in ``line_points`` using a
/// brute-force algorithm. Returns a tuple of the closest ``line_points`` locations and
/// their associated distances.
///
/// :param np.ndarray line_points: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_LINE_POINTS, 2)``
/// :param np.ndarray point_cloud: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_POINT_CLOUD_POINTS, 2)``
/// :param bool parallel: enable parallel processing for the dataset. If you have more than 2,000 line points and 2,000 point cloud points this may be useful.
///
/// this function returns  a ``(NUM_POINT_CLOUD_POINTS, 2)`` shaped array of the points where
/// each ``i`` th row of the returned array is a row of ``line_points`` that is closest to the
/// ``i`` th row of ``point_cloud``
///
/// Example:
///
/// .. code-block::
///     
///     import toha_nearest_neighbor
///     import numpy as np
///
///     line_points = np.array(
///         [
///             [0.0, 0.0],
///             [1.0, 1.0],
///             [2.0, 2.0],
///         ]
///     )
///
///     point_cloud = np.array(
///         [
///             [0.1, -0.1], #closest to the 0-th index of line_points rows
///             [2.2, 3.0], # closest to the 2-nd index of line_points rows
///         ]
///     )
///
///     (closest_line_points, distances) = toha_nearest_neighbor.kd_tree_location(line_points, point_cloud, parallel = True)
///
///     # [[0. 0.]
///     #  [2. 2.]]
///     print(closest_line_points)
///     # [0.02 1.04]
///     print(distances)
///
fn kd_tree_location<'a>(
    py: Python<'a>,
    line_points: PyReadonlyArray2<'a, f64>,
    point_cloud: PyReadonlyArray2<'a, f64>,
    parallel: bool,
) -> (&'a PyArray2<f64>, &'a PyArray1<f64>) {
    let location_and_distance = if parallel {
        macros::dimension_expansion!(
            super::kd_tree_location_par(line_points.as_array(), point_cloud.as_array()),
            15
        )
    } else {
        macros::dimension_expansion!(
            super::kd_tree_location(line_points.as_array(), point_cloud.as_array()),
            15
        )
    };

    let loc = PyArray2::from_owned_array(py, location_and_distance.location);
    let distance = PyArray1::from_owned_array(py, location_and_distance.distance);

    (loc, distance)
}

#[pyfunction]
#[pyo3(signature = (line_points, point_cloud, parallel = false))]
/// For every point in ``point_cloud``, find it's nearest neighbor in ``line_points`` using a
/// kd-tree algorithm. Returns a tuple of indicies of the closest ``line_points`` rows and
/// their associated distances.
///
/// :param np.ndarray line_points: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_LINE_POINTS, 2)``
/// :param np.ndarray point_cloud: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_POINT_CLOUD_POINTS, 2)``
/// :param bool parallel: enable parallel processing for the dataset. If you have more than 2,000 line points and 2,000 point cloud points this may be useful.
///
/// this function returns  a ``(NUM_POINT_CLOUD_POINTS, 2)`` shaped array of the points where
/// each ``i`` th row of the returned array is a row of ``line_points`` that is closest to the
/// ``i`` th row of ``point_cloud``
///
/// Example:
///
/// .. code-block::
///     
///     import toha_nearest_neighbor
///     import numpy as np
///
///     line_points = np.array(
///         [
///             [0.0, 0.0],
///             [1.0, 1.0],
///             [2.0, 2.0],
///         ]
///     )
///
///     point_cloud = np.array(
///         [
///             [0.1, -0.1], #closest to the 0-th index of line_points rows
///             [2.2, 3.0], # closest to the 2-nd index of line_points rows
///         ]
///     )
///
///     indicies, distances = toha_nearest_neighbor.kd_tree_index(line_points, point_cloud, parallel = True)
///     # [0 2]
///     print(indicies)
///     # [0.02 1.04]
///     print(distances)
///
fn kd_tree_index<'a>(
    py: Python<'a>,
    line_points: PyReadonlyArray2<'a, f64>,
    point_cloud: PyReadonlyArray2<'a, f64>,
    parallel: bool,
) -> (&'a PyArray1<usize>, &'a PyArray1<f64>) {
    let index_and_distance = if parallel {
        macros::dimension_expansion!(
            super::kd_tree_index_par(line_points.as_array(), point_cloud.as_array()),
            15
        )
    } else {
        macros::dimension_expansion!(
            super::kd_tree_index(line_points.as_array(), point_cloud.as_array()),
            15
        )
    };

    let index = PyArray1::from_owned_array(py, index_and_distance.index);
    let distance = PyArray1::from_owned_array(py, index_and_distance.distance);

    (index, distance)
}

/// This module is implemented in Rust.
#[pymodule]
fn toha_nearest_neighbor(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brute_force_location, m)?)?;
    m.add_function(wrap_pyfunction!(brute_force_index, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pybind::kd_tree_location, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pybind::kd_tree_index, m)?)?;
    Ok(())
}
