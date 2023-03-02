import toha_nearest_neighbor as toha
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Tuple
import time
from statistics import mean
import math
from scipy.spatial import KDTree

DIM = 2

def create_data(line_ct: int, point_ct: int) -> Tuple[np.ndarray, np.ndarray]:
    line_pts = np.random.rand(line_ct, DIM)
    cloud_pts = np.random.rand(point_ct, DIM)

    return line_pts, cloud_pts 

def bench_python_brute(lines: np.ndarray, cloud: np.ndarray):
    out_array = np.zeros(cloud.shape[0])

    for cloud_row in range(cloud.shape[0]):
        cloud_row_val = cloud[cloud_row, :]

        minimum = math.inf
        minimum_idx = 0;

        for line_row in range(lines.shape[0]):
            line_row_val = lines[line_row, :]

            distance = 0

            for i in range(lines.shape[1]):
                distance += (line_row_val[i] - cloud_row_val[i])**2

            if distance < minimum:
                minimum_idx = line_row
                minimum = distance

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

def bench_scipy(lines: np.ndarray, cloud: np.ndarray, compact:bool):
    kd = KDTree(lines, compact_nodes = compact)
    kd.query(cloud)

def bench_rust_brute(lines: np.ndarray, cloud: np.ndarray, parallel: bool):
    toha.brute_force_index(lines, cloud, parallel)

    return

def bench_rust_kd(lines: np.ndarray, cloud: np.ndarray, parallel: bool):
    toha.kd_tree_index(lines, cloud, parallel)

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
    scipy_kd = []
    scipy_kd_compact = []
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
        if line_size <= 5_000:
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

        #scipy kd
        l = lambda : bench_scipy(lines, clouds, False)
        bench_helper(l, scipy_kd)

        #scipy kd compact
        l = lambda : bench_scipy(lines, clouds, True)
        bench_helper(l, scipy_kd_compact)

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

    style_brute = "solid"
    style_kd = "dashed"

    ax.plot(python_xs, python_brute, label = "python brute", color = "black", linestyle = style_brute)
    ax.plot(xs, numpy_brute, label = "numpy brute", color = "blue", linestyle = style_brute)

    ax.plot(xs, scikit_brute, label = "scikit brute", color = "green", linestyle = style_brute)
    ax.plot(xs, scikit_kd, label = "scikit kd", color = "green", linestyle = style_kd)

    ax.plot(xs, scipy_kd, label = "scipy kd", color = "cyan", linestyle = style_kd)
    ax.plot(xs, scipy_kd_compact, label = "scipy kd compact", color = "grey", linestyle = style_kd)

    ax.plot(xs, rust_brute, label = "rust brute", color = "red", linestyle=style_brute)
    ax.plot(xs, rust_kd, label = "rust kd", color = "red", linestyle=style_kd)

    ax.plot(xs, rust_brute_par, label = "rust brute parallel", color = "orange", linestyle = style_brute)
    ax.plot(xs, rust_kd_par, label = "rust kd parallel", color = "orange", linestyle = style_kd)

    plt.legend()

    plt.savefig("./static/benchmarks.png", bbox_inches="tight")


bench_all()
