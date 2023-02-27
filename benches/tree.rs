use toha::kd_tree_index;
use toha::kd_tree_index_par;
use toha::kd_tree_location;
use toha::kd_tree_location_par;
use toha_nearest_neighbor as toha;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

//const LINE_POINTS : [usize; 3] = [1,5,20];
//const CLOUD_POINTS : [usize; 3] = [1,10,20];
const LINE_POINTS: [usize; 2] = [1, 5];
const CLOUD_POINTS: [usize; 2] = [1, 10];

fn create_data(line_length: usize, points_length: usize) -> (Array2<f64>, Array2<f64>) {
    let lines = ndarray::Array2::random((line_length, 2), Uniform::new(0.0, 10.0));
    let points = ndarray::Array2::random((points_length, 2), Uniform::new(0.0, 10.0));

    (lines, points)
}

fn serial(c: &mut Criterion) {
    for (line_ct, cloud_ct) in LINE_POINTS.into_iter().zip(CLOUD_POINTS) {
        let (lines, points) = create_data(line_ct * 1000, cloud_ct * 1000);
        let name = format!(
            "kd tree location | serial | {line_ct}k line points |  {cloud_ct}k cloud points"
        );

        c.bench_function(&name, |b| {
            b.iter(|| black_box(kd_tree_location::<2>(lines.view(), points.view())))
        });

        let name =
            format!("kd tree index| serial | {line_ct}k line points |  {cloud_ct}k cloud points");

        c.bench_function(&name, |b| {
            b.iter(|| black_box(kd_tree_index::<2>(lines.view(), points.view())))
        });
    }
}

fn parallel(c: &mut Criterion) {
    for (line_ct, cloud_ct) in LINE_POINTS.into_iter().zip(CLOUD_POINTS) {
        let (lines, points) = create_data(line_ct * 1000, cloud_ct * 1000);
        let name = format!(
            "kd tree location | parallel | {line_ct}k line points |  {cloud_ct}k cloud points"
        );

        c.bench_function(&name, |b| {
            b.iter(|| {
                black_box(kd_tree_location_par::<2>(lines.view(), points.view()));
            })
        });

        let name = format!(
            "kd tree index | parallel | {line_ct}k line points |  {cloud_ct}k cloud points"
        );

        c.bench_function(&name, |b| {
            b.iter(|| {
                black_box(kd_tree_index_par::<2>(lines.view(), points.view()));
            })
        });
    }
}

criterion_group!(benches, serial, parallel);
criterion_main!(benches);
