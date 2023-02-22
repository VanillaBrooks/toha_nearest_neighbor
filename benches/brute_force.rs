use toha_nearest_neighbor::brute_force;
use toha_nearest_neighbor::brute_force_par;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn create_data(line_length: usize, points_length: usize) -> (Array2<f64>, Array2<f64>) {
    let lines = ndarray::Array2::random((line_length, 2), Uniform::new(0.0, 10.0));
    let points = ndarray::Array2::random((points_length, 2), Uniform::new(0.0, 10.0));

    (lines, points)
}

fn serial(c: &mut Criterion) {
    let lines_pts = [1, 5, 20];
    let clout_pts = [1, 10, 20];

    for (line_ct, cloud_ct) in lines_pts.into_iter().zip(clout_pts) {
        let (lines, points) = create_data(line_ct * 1000, cloud_ct * 1000);
        let name =
            format!("brute force | serial | {line_ct}k line points |  {cloud_ct}k cloud points");

        c.bench_function(&name, |b| {
            b.iter(|| black_box(brute_force(lines.view(), points.view())))
        });
    }
}

fn parallel(c: &mut Criterion) {
    let lines_pts = [1, 5, 20];
    let clout_pts = [1, 10, 20];

    for (line_ct, cloud_ct) in lines_pts.into_iter().zip(clout_pts) {
        let (lines, points) = create_data(line_ct * 1000, cloud_ct * 1000);
        let name =
            format!("brute force | parallel | {line_ct}k line points |  {cloud_ct}k cloud points");

        c.bench_function(&name, |b| {
            b.iter(|| {
                black_box(brute_force_par(lines.view(), points.view()));
            })
        });
    }
}

criterion_group!(benches, serial, parallel);
criterion_main!(benches);
