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
    let (lines, points) = create_data(40_000, 20_000);

    c.bench_function("serial | 40k line points | 20k match points", |b| {
        b.iter(|| {
            brute_force(lines.view(), points.view());
        })
    });

    let (lines, points) = create_data(1000, 2000);

    c.bench_function("serial | 1k line points | 2k match points", |b| {
        b.iter(|| {
            brute_force(lines.view(), points.view());
        })
    });
}

fn parallel(c: &mut Criterion) {
    let (lines, points) = create_data(40_000, 20_000);

    c.bench_function("parallel | 40k line points | 20k match points", |b| {
        b.iter(|| {
            brute_force_par(lines.view(), points.view());
        })
    });

    let (lines, points) = create_data(1000, 2000);

    c.bench_function("parallel | 1k line points | 2k match points", |b| {
        b.iter(|| {
            brute_force(lines.view(), points.view());
        })
    });
}

criterion_group!(benches, serial, parallel);
criterion_main!(benches);
