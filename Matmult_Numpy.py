import time      # For time.perf_counter() to measure execution time
import random    # Though NumPy's random is preferred for arrays, this is still needed if you use random.seed
import sys       # Potentially useful for recursion limits, but likely not strictly needed
import numpy as np # Import NumPy

# Constants for Benchmarking
# Extended matrix sizes up to 1024x1024 (powers of 2)
MATRIX_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,2048,4096,8192]
NUM_ITERATIONS = 10             # Number of runs for each matrix size

# --- Helper Functions (NumPy-based) ---

def create_matrix_np(n, random_fill=False):
    """
    Creates an n x n NumPy array, initialized with zeros or random values.
    """
    if random_fill:
        # np.random.uniform creates an array of random floats
        matrix = np.random.uniform(0.0, 10.0, size=(n, n))
    else:
        matrix = np.zeros((n, n), dtype=np.float64) # Create n x n array of zeros
    return matrix

def add_matrices_np(A, B):
    """
    Performs element-wise addition of two square NumPy matrices A and B.
    Returns C = A + B.
    """
    return A + B

def subtract_matrices_np(A, B):
    """
    Performs element-wise subtraction of two square NumPy matrices A and B.
    Returns C = A - B.
    """
    return A - B

def split_matrix_np(parent_matrix, r_start, c_start, size):
    """
    Extracts a square sub-matrix from a parent NumPy matrix using slicing.
    r_start: starting row index in parent_matrix
    c_start: starting column index in parent_matrix
    size: dimension of the square sub-matrix
    """
    # NumPy slicing syntax: matrix[row_start:row_end, col_start:col_end]
    sub_matrix = parent_matrix[r_start : r_start + size, c_start : c_start + size]
    return sub_matrix

def join_matrices_np(C11, C12, C21, C22):
    """
    Combines four square NumPy sub-matrices into a single larger matrix.
    """
    # np.block efficiently constructs a matrix from sub-blocks
    C = np.block([[C11, C12], [C21, C22]])
    return C

def naive_matrix_multiply_np(A, B):
    """
    Performs standard (naive) matrix multiplication C = A @ B using NumPy's optimized operator.
    """
    return np.dot(A, B)

# --- Strassen's Algorithm Implementation (NumPy-based) ---

def strassens_multiply_np(A, B):
    """
    Implements Strassen's Matrix Multiplication algorithm using NumPy arrays and operations.
    Recursively multiplies two square matrices A and B.
    Assumes matrices have dimensions that are powers of 2.
    """
    n = A.shape[0] # Get dimension from NumPy array shape

    # Base case: If matrix size is 1x1, perform scalar multiplication
    # For NumPy, A * B on 1x1 arrays is still an element-wise multiplication
    # The result is a 1x1 array, which is correct for consistency.
    if n == 1:
        return A * B
    
   
    half_n = n // 2

    # Divide matrices A and B into 4 equally sized sub-matrices using split_matrix_np
    A11 = split_matrix_np(A, 0, 0, half_n)
    A12 = split_matrix_np(A, 0, half_n, half_n)
    A21 = split_matrix_np(A, half_n, 0, half_n)
    A22 = split_matrix_np(A, half_n, half_n, half_n)

    B11 = split_matrix_np(B, 0, 0, half_n)
    B12 = split_matrix_np(B, 0, half_n, half_n)
    B21 = split_matrix_np(B, half_n, 0, half_n)
    B22 = split_matrix_np(B, half_n, half_n, half_n)

    # Step 1: Compute the 7 intermediate products (P matrices) recursively
    P1 = strassens_multiply_np(A11, subtract_matrices_np(B12, B22))
    P2 = strassens_multiply_np(add_matrices_np(A11, A12), B22)
    P3 = strassens_multiply_np(add_matrices_np(A21, A22), B11)
    P4 = strassens_multiply_np(A22, subtract_matrices_np(B21, B11))
    P5 = strassens_multiply_np(add_matrices_np(A11, A22), add_matrices_np(B11, B22))
    P6 = strassens_multiply_np(subtract_matrices_np(A21, A11), add_matrices_np(B11, B12))
    P7 = strassens_multiply_np(subtract_matrices_np(A12, A22), add_matrices_np(B21, B22))

    # Step 2: Combine the P matrices to get the four resulting C sub-matrices
    C11 = add_matrices_np(subtract_matrices_np(add_matrices_np(P5, P4), P2), P7)
    C12 = add_matrices_np(P1, P2)
    C21 = add_matrices_np(P3, P4)
    C22 = subtract_matrices_np(subtract_matrices_np(add_matrices_np(P5, P1), P3), P6)

    # Step 3: Join the four C sub-matrices into the final result matrix C
    return join_matrices_np(C11, C12, C21, C22)

# Main Execution and Benchmarking 
if __name__ == "__main__":
    print("Benchmarking Matrix Multiplication in Python with NumPy")
    print(f"Number of Iterations per test size: {NUM_ITERATIONS}")
    print("Matrix Size (N) | Total Time Naive (s) | Avg Time Naive (ns) | "
          "Total Time Strassen (s) | Avg Time Strassen (ns)")

    for n in MATRIX_SIZES:
        total_naive_sec = 0.0
        total_strassen_sec = 0.0
        strassen_executed = True

        for _ in range(NUM_ITERATIONS):
            A_np = create_matrix_np(n, random_fill=True)
            B_np = create_matrix_np(n, random_fill=True)

            # ---- NumPy Naive ----
            start = time.perf_counter()
            C_naive = naive_matrix_multiply_np(A_np, B_np)
            end = time.perf_counter()
            total_naive_sec += (end - start)
            _ = np.sum(C_naive)

            # ---- NumPy Strassen (only for n <= 64) ----
            if n <= 64:
                start = time.perf_counter()
                C_strassen = strassens_multiply_np(A_np, B_np)
                end = time.perf_counter()
                total_strassen_sec += (end - start)
                _ = np.sum(C_strassen)
            else:
                strassen_executed = False
                break

        avg_naive_ns = (total_naive_sec / NUM_ITERATIONS) * 1e9

        if not strassen_executed:
            print(f"{n:<14} | {total_naive_sec:<20.6f} | {int(avg_naive_ns):<20} | "
                  f"N/A                   | N/A")
        else:
            avg_strassen_ns = (total_strassen_sec / NUM_ITERATIONS) * 1e9
            print(f"{n:<14} | {total_naive_sec:<20.6f} | {int(avg_naive_ns):<20} | "
                  f"{total_strassen_sec:<21.6f} | {int(avg_strassen_ns)}")

