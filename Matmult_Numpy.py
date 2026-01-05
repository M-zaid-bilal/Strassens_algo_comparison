import time
import random
import sys
import numpy as np

MATRIX_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
NUM_ITERATIONS = 10

def create_matrix_np(n, random_fill=False):
    if random_fill:
        return np.random.uniform(0.0, 10.0, size=(n, n))
    else:
        return np.zeros((n, n), dtype=np.float64)

def naive_matrix_multiply_np(A, B):
    return np.dot(A, B)

if __name__ == "__main__":
    print("Numpy Based Optimized Matmult - Benchmarking Matrix Multiplication with NumPy")
    print(f"Number of Iterations per test size: {NUM_ITERATIONS}")
    print("Matrix Size (N) | NumPy Matmult Total (s) | NumPy Matmult Avg (ns)")

    for n in MATRIX_SIZES:
        total_duration_np_naive = 0.0

        for _ in range(NUM_ITERATIONS):
            A_np = create_matrix_np(n, random_fill=True)
            B_np = create_matrix_np(n, random_fill=True)

            start_time_np_naive = time.perf_counter()
            C_np_naive = naive_matrix_multiply_np(A_np, B_np)
            end_time_np_naive = time.perf_counter()
            total_duration_np_naive += (end_time_np_naive - start_time_np_naive)

            # touch result to avoid lazy optimizations (and to ensure memory use)
            _ = np.sum(C_np_naive)

        avg_np_naive_ns = (total_duration_np_naive / NUM_ITERATIONS) * 1e9

        print(f"{n:>6}         | {total_duration_np_naive:12.6f}           | {avg_np_naive_ns:16.6f}")

    # Single large-run: run naive for largest configured size
    n_large = MATRIX_SIZES[-1]
    A_large_np = create_matrix_np(n_large, random_fill=True)
    B_large_np = create_matrix_np(n_large, random_fill=True)

    print(f"\nSingle run for {n_large}x{n_large} (1 iteration):")

    start_time_large_np_naive = time.perf_counter()
    C_large_np_naive = naive_matrix_multiply_np(A_large_np, B_large_np)
    end_time_large_np_naive = time.perf_counter()
    time_large_np_naive = end_time_large_np_naive - start_time_large_np_naive
    _ = np.sum(C_large_np_naive)
    avg_large_np_naive_ns = time_large_np_naive * 1e9
    print(f"NumPy Matmult  {time_large_np_naive:.6f} seconds | Avg: {avg_large_np_naive_ns:.6f} ns")
