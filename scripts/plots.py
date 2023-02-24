import toha_nearest_neighbor as toha
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Tuple
import time
from statistics import mean
import math

def create_data(line_ct: int, point_ct: int) -> Tuple[np.ndarray, np.ndarray]:
    line_pts = np.random.rand(line_ct, 2)
    cloud_pts = np.random.rand(point_ct, 2)

    return line_pts, cloud_pts 

def bench_python_brute(lines: np.ndarray, cloud: np.ndarray):
    out_array = np.zeros(cloud.shape[0])

    for cloud_row in range(cloud.shape[0]):
        cloud_row_val = cloud[cloud_row, :]

        minimum = math.inf
        minimum_idx = 0;

        for line_row in range(lines.shape[0]):
            line_row_val = lines[line_row, :]
            candidate_min = (line_row_val[0] - cloud_row_val[0])**2 + (line_row_val[1] - cloud_row_val[1])**2

            if candidate_min < minimum:
                minimum_idx = line_row
                minimum = candidate_min

        out_array[cloud_row] = minimum_idx

    return out_array

def bench_numpy_brute(lines: np.ndarray, cloud: np.ndarray):
    out_array = np.zeros(cloud.shape[0])

    for row in range(cloud.shape[0]):
        diff = lines - cloud[row, :]
        mag = np.linalg.norm(diff, axis=1)

        min_distance = np.argmin(mag)
        out_array[row] = min_distance

    return out_array

def bench_scikit(lines: np.ndarray, cloud: np.ndarray, algorithm: str):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm=algorithm).fit(lines)
    distances, indices = nbrs.kneighbors(cloud)

    return

def bench_rust_brute(lines: np.ndarray, cloud: np.ndarray, parallel: bool):
    toha.brute_force(lines, cloud, parallel)

    return

def bench_rust_kd(lines: np.ndarray, cloud: np.ndarray, parallel: bool):
    toha.kd_tree(lines, cloud, parallel)

    return

def bench(fn, times: int) -> float:
    runtimes = []

    for _ in range(times):
        start = time.time()
        fn()
        end = time.time()
        runtimes.append(end - start)

    return mean(runtimes)

def bench_helper(fn, output_list: list[float]):
    out_mean = bench(fn, 10)

    print(f"mean runtime was {out_mean}")

    # convert the mean time to ms
    output_list.append(out_mean)


def bench_all():
    line_sizes = [100, 500, 1000, 5_000, 10_000, 15_000, 20_000, 30_000]
    #line_sizes = [100, 500, 1000]
    cloud_sizes = line_sizes.copy()

    python_brute= []
    numpy_brute= []
    scikit_brute = []
    scikit_kd = []
    rust_brute = []
    rust_kd = []
    rust_brute_par = []
    rust_kd_par = []

    xs = []
    python_xs = []

    for (line_size, cloud_size) in zip(line_sizes, cloud_sizes):
        lines, clouds = create_data(line_size, cloud_size)

        xs.append(line_size * cloud_size)

        # python brute
        if line_size <= 10_000:
            python_xs.append(line_size * cloud_size)

            l = lambda : bench_python_brute(lines, clouds)
            bench_helper(l, python_brute)

        # numpy brute
        l = lambda : bench_numpy_brute(lines, clouds)
        bench_helper(l, numpy_brute)

        # scikit brute
        l = lambda : bench_scikit(lines, clouds, "brute")
        bench_helper(l, scikit_brute)

        # scikit kd
        l = lambda : bench_scikit(lines, clouds, "kd_tree")
        bench_helper(l, scikit_kd)

        # rust brute serial
        l = lambda : bench_rust_brute(lines, clouds, False)
        bench_helper(l, rust_brute)

        # rust kd serial
        l = lambda : bench_rust_kd(lines, clouds, False)
        bench_helper(l, rust_kd)

        # rust brute parallel
        l = lambda : bench_rust_brute(lines, clouds, True)
        bench_helper(l, rust_brute_par)

        # rust kd parallel
        l = lambda : bench_rust_kd(lines, clouds, True)
        bench_helper(l, rust_kd_par)

        print(f"finished size {line_size} | {cloud_size}")

    fig = plt.figure(figsize = (8, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("total point size (num neighbors * num point cloud)")
    ax.set_ylabel("runtime [s]")

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.plot(python_xs, python_brute, label = "python brute", color = "black", linestyle = "solid")
    ax.plot(xs, numpy_brute, label = "numpy brute", color = "red", linestyle = "dashdot")
    ax.plot(xs, scikit_brute, label = "scikit brute", color = "red")
    ax.plot(xs, scikit_kd, label = "scikit kd", color = "blue")
    ax.plot(xs, rust_brute, label = "rust brute", color = "red", linestyle="dotted")
    ax.plot(xs, rust_kd, label = "rust kd", color = "blue", linestyle="dotted")
    ax.plot(xs, rust_brute_par, label = "rust brute parallel", color = "red", linestyle = "dashed")
    ax.plot(xs, rust_kd_par, label = "rust kd parallel", color = "blue", linestyle = "dashed")

    plt.legend()

    plt.savefig("./static/benchmarks.png", bbox_inches="tight")


bench_all()
