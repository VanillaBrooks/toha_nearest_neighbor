use numpy::PyArray2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (line_points, point_cloud, parallel = false))]
/// For every point in ``point_cloud``, find it's nearest neighbor in ``line_points`` using a
/// brute-force algorithm
///
/// :param np.ndarray line_points: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_LINE_POINTS, 2)``
/// :param np.ndarray point_cloud: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_POINT_CLOUD_POINTS, 2)``
/// :param bool parallel: enable parallel processing for the dataset. If you have more than 2,000 line points and 2,000 point cloud points this may be useful.
///
/// this function returns  a ``(NUM_POINT_CLOUD_POINTS, 2)`` shaped array of the points where
/// each ``i``th row of the returned array is a row of ``line_points`` that is closest to the
/// ``i``th row of ``point_cloud``
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
///     # output: 
///     # [[0. 0.]
///     #  [2. 2.]]
///     toha_nearest_neighbor.brute_force(line_points, point_cloud)
fn brute_force<'a>(
    py: Python<'a>,
    line_points: PyReadonlyArray2<'_, f64>,
    point_cloud: PyReadonlyArray2<'_, f64>,
    parallel: bool,
) -> &'a PyArray2<f64> {
    let arr = if parallel {
        super::brute_force_par(line_points.as_array(), point_cloud.as_array())
    } else {
        super::brute_force(line_points.as_array(), point_cloud.as_array())
    };

    PyArray2::from_owned_array(py, arr)
}

#[pyfunction]
#[pyo3(signature = (line_points, point_cloud, parallel = false))]
/// For every point in ``point_cloud``, find it's nearest neighbor in ``line_points`` using a
/// kd-tree algorithm
///
/// :param np.ndarray line_points: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_LINE_POINTS, 2)``
/// :param np.ndarray point_cloud: a 2D numpy array with each point being described by a row, and columns of coordinates of that point. Array should be in the shape ``(NUM_POINT_CLOUD_POINTS, 2)``
/// :param bool parallel: enable parallel processing for the dataset. If you have more than 2,000 line points and 2,000 point cloud points this may be useful.
///
/// this function returns  a ``(NUM_POINT_CLOUD_POINTS, 2)`` shaped array of the points where
/// each ``i``th row of the returned array is a row of ``line_points`` that is closest to the
/// ``i``th row of ``point_cloud``
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
///     # output: 
///     # [[0. 0.]
///     #  [2. 2.]]
///     toha_nearest_neighbor.kd_tree(line_points, point_cloud, parallel = True)
fn kd_tree<'a>(
    py: Python<'a>,
    line_points: PyReadonlyArray2<'a, f64>,
    point_cloud: PyReadonlyArray2<'a, f64>,
    parallel: bool,
) -> &'a PyArray2<f64> {
    let arr = if parallel {
        super::kd_tree_par(line_points.as_array(), point_cloud.as_array())
    } else {
        super::kd_tree(line_points.as_array(), point_cloud.as_array())
    };

    PyArray2::from_owned_array(py, arr)
}

/// This module is implemented in Rust.
#[pymodule]
fn toha_nearest_neighbor(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brute_force, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pybind::kd_tree, m)?)?;
    Ok(())
}
