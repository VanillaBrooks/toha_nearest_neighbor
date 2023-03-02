use kd_tree::KdIndexTreeN;
use kd_tree::KdTreeN;

use ndarray::ArrayView2;
use ndarray::Axis;

use rayon::prelude::*;

use super::FromShapeIter;
use super::IndexAndDistance;
use super::LocationAndDistance;
use super::SingleIndexDistance;
use super::SinglePointDistance;
use super::SinglePointDistanceRef;

/// KD-Tree solution to the nearest neighbor problem with serial iteration
///
/// ## Parameters
///
/// `line_points` is the 2D array of datapoints that are candidates for the 2D array of points in
/// `points_to_match`. Essentially, every row of `points_to_match` contains two columns (x,y
/// location floats) that will be matched against all rows of `line_points` (in the same format)
/// to find the minimum distance.
///
/// Returns information on the location of the closest point and its distance.
pub fn kd_tree_location<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> LocationAndDistance
where
    [f64; DIM]: kd_tree::KdPoint<DIM>,
    <[f64; DIM] as kd_tree::KdPoint<DIM>>::Scalar: num_traits::Float + Into<f64>,
{
    let line_points = to_kdtree_vector::<DIM>(line_points);
    let kdtree = assemble_location_tree::<DIM>(line_points);

    let point_iter = points_to_match.axis_iter(Axis(0)).map(|point| {
        let array_point = super::copy_to_array::<DIM>(point);

        let item = kdtree.nearest(&array_point).unwrap();

        SinglePointDistanceRef {
            point: item.item,
            distance: item.squared_distance.into(),
        }
    });

    LocationAndDistance::from_shape_iter(point_iter, points_to_match.dim())
}

/// KD-Tree solution to the nearest neighbor problem with serial iteration
///
/// ## Parameters
///
/// `line_points` is the 2D array of datapoints that are candidates for the 2D array of points in
/// `points_to_match`. Essentially, every row of `points_to_match` contains two columns (x,y
/// location floats) that will be matched against all rows of `line_points` (in the same format)
/// to find the minimum distance.
///
/// Returns information on the row-index of the closest point and its distance.
pub fn kd_tree_index<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> IndexAndDistance
where
    [f64; DIM]: kd_tree::KdPoint<DIM>,
    <[f64; DIM] as kd_tree::KdPoint<DIM>>::Scalar: num_traits::Float + Into<f64>,
{
    let line_points = to_kdtree_vector::<DIM>(line_points);
    let kdtree = assemble_index_tree::<DIM>(&line_points);

    let point_iter = points_to_match.axis_iter(Axis(0)).map(|point| {
        let array_point = super::copy_to_array::<DIM>(point);

        let item = kdtree.nearest(&array_point).unwrap();

        SingleIndexDistance {
            index: *item.item,
            distance: item.squared_distance.into(),
        }
    });

    super::FromShapeIter::<DIM, _>::from_shape_iter(point_iter, points_to_match.dim())
    //IndexAndDistance::from_shape_iter(point_iter, points_to_match.dim())
}

/// KD-Tree solution to the nearest neighbor problem with parallel iteration
///
/// ## Parameters
///
/// `line_points` is the 2D array of datapoints that are candidates for the 2D array of points in
/// `points_to_match`. Essentially, every row of `points_to_match` contains two columns (x,y
/// location floats) that will be matched against all rows of `line_points` (in the same format)
/// to find the minimum distance.
///
/// Returns information on the location of the closest point and its distance.
pub fn kd_tree_location_par<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> LocationAndDistance
where
    [f64; DIM]: kd_tree::KdPoint<DIM>,
    <[f64; DIM] as kd_tree::KdPoint<DIM>>::Scalar: num_traits::Float + Into<f64>,
{
    let line_points = to_kdtree_vector::<DIM>(line_points);
    let kdtree = assemble_location_tree::<DIM>(line_points);

    let points_vec: Vec<_> = points_to_match
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|point| {
            let array_point = super::copy_to_array::<DIM>(point);

            let item = kdtree.nearest(&array_point).unwrap();

            SinglePointDistance {
                point: *item.item,
                distance: item.squared_distance.into(),
            }
        })
        // this allocation is not ideal here, but it seems to be unavoidable
        .collect();

    LocationAndDistance::from_shape_iter(points_vec, points_to_match.dim())
}

/// KD-Tree solution to the nearest neighbor problem with parallel iteration
///
/// ## Parameters
///
/// `line_points` is the 2D array of datapoints that are candidates for the 2D array of points in
/// `points_to_match`. Essentially, every row of `points_to_match` contains two columns (x,y
/// location floats) that will be matched against all rows of `line_points` (in the same format)
/// to find the minimum distance.
///
/// Returns information on the row-index of the closest point and its distance.
pub fn kd_tree_index_par<const DIM: usize>(
    line_points: ArrayView2<'_, f64>,
    points_to_match: ArrayView2<'_, f64>,
) -> IndexAndDistance
where
    [f64; DIM]: kd_tree::KdPoint<DIM>,
    <[f64; DIM] as kd_tree::KdPoint<DIM>>::Scalar: num_traits::Float + Into<f64>,
{
    let line_points = to_kdtree_vector::<DIM>(line_points);
    let kdtree = assemble_index_tree::<DIM>(&line_points);

    let points_vec: Vec<_> = points_to_match
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|point| {
            let array_point = super::copy_to_array::<DIM>(point);

            let item = kdtree.nearest(&array_point).unwrap();

            SingleIndexDistance {
                index: *item.item,
                distance: item.squared_distance.into(),
            }
        })
        // this allocation is not ideal here, but it seems to be unavoidable
        .collect();

    //IndexAndDistance::from_shape_iter(points_vec, points_to_match.dim())
    super::FromShapeIter::<DIM, _>::from_shape_iter(points_vec, points_to_match.dim())
}

fn to_kdtree_vector<const DIM: usize>(line_points: ArrayView2<'_, f64>) -> Vec<[f64; DIM]> {
    line_points
        .axis_iter(Axis(0))
        .map(|point| {
            let array_point = super::copy_to_array::<DIM>(point);

            array_point
        })
        .collect()
}

fn assemble_index_tree<const DIM: usize>(
    line_points: &[[f64; DIM]],
) -> KdIndexTreeN<[f64; DIM], DIM>
where
    [f64; DIM]: kd_tree::KdPoint<DIM>,
    <[f64; DIM] as kd_tree::KdPoint<DIM>>::Scalar: num_traits::Float,
{
    KdIndexTreeN::build_by_ordered_float(line_points)
}

fn assemble_location_tree<const DIM: usize>(
    line_points: Vec<[f64; DIM]>,
) -> KdTreeN<[f64; DIM], DIM>
where
    [f64; DIM]: kd_tree::KdPoint<DIM>,
    <[f64; DIM] as kd_tree::KdPoint<DIM>>::Scalar: num_traits::Float,
{
    KdTreeN::build_by_ordered_float(line_points)
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn parallel_serial_same() {
        let lines = Array2::random((10000, 2), Uniform::new(0.0, 10.0));
        let points = Array2::random((3000, 2), Uniform::new(0.0, 10.0));

        let kd_brute = kd_tree_location::<2>(lines.view(), points.view());
        let kd_par = kd_tree_location_par::<2>(lines.view(), points.view());

        assert_eq!(kd_brute, kd_par);
    }

    #[test]
    fn kdtree_brute_force_same() {
        let lines = Array2::random((100, 2), Uniform::new(0.0, 10.0));
        let points = Array2::random((100, 2), Uniform::new(0.0, 10.0));

        let out_kd = kd_tree_location::<2>(lines.view(), points.view());
        let out_brute = crate::brute_force_location::<2>(lines.view(), points.view());

        assert_eq!(out_kd, out_brute);
    }
}
