import time
import random
import sys


iterations=10
mat_size=[2,4,8,16,32,64]

def createMatrix(n, random_fill=False):
     matrix = [[0.0 for _ in range(n)] for _ in range(n)]
     if random_fill:
         for i in range(n):
             for j in range(n):
                 matrix[i][j] = random.uniform(0.0, 10.0) 
     return matrix

def add_matrices(A, B):
    n = len(A)
    C = createMatrix(n) 
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C

def subtract_matrices(A,B):
    n=len(A)
    C=createMatrix(n)
    for i in range(n):
        for j in range(n):
            C[i][j]=A[i][j]-B[i][j]
    return C

def split_matrix(parent_matrix, r_start, c_start, size):
    sub_matrix = createMatrix(size)
    for i in range(size):
        for j in range(size):
            sub_matrix[i][j] = parent_matrix[r_start + i][c_start + j]
    return sub_matrix

def join_matrices(C11, C12, C21, C22):
   
    half_n = len(C11)
    n = half_n * 2    # Dimension of the combined matrix
    C = createMatrix(n)

    for i in range(half_n):
        for j in range(half_n):
            C[i][j] = C11[i][j]                  
            C[i][j + half_n] = C12[i][j]          
            C[i + half_n][j] = C21[i][j]          
            C[i + half_n][j + half_n] = C22[i][j] 
    return C

def naive_matrix_multiply(A, B):
    n = len(A)
    C = createMatrix(n) 
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def strassens_Multiply(A, B):
    n = len(A)

    # Base case: If matrix size is 1x1, perform scalar multiplication
    if n == 1:
        C = createMatrix(1)
        C[0][0] = A[0][0] * B[0][0]
        return C

    half_n = n // 2

    A11 = split_matrix(A, 0, 0, half_n)
    A12 = split_matrix(A, 0, half_n, half_n)
    A21 = split_matrix(A, half_n, 0, half_n)
    A22 = split_matrix(A, half_n, half_n, half_n)

    B11 = split_matrix(B, 0, 0, half_n)
    B12 = split_matrix(B, 0, half_n, half_n)
    B21 = split_matrix(B, half_n, 0, half_n)
    B22 = split_matrix(B, half_n, half_n, half_n)



    P1 = strassens_Multiply(A11, subtract_matrices(B12, B22))
    P2 = strassens_Multiply(add_matrices(A11, A12), B22)
    P3 = strassens_Multiply(add_matrices(A21, A22), B11)
    P4 = strassens_Multiply(A22, subtract_matrices(B21, B11))
    P5 = strassens_Multiply(add_matrices(A11, A22), add_matrices(B11, B22))
    P6 = strassens_Multiply(subtract_matrices(A21, A11), add_matrices(B11, B12))
    P7 = strassens_Multiply(subtract_matrices(A12, A22), add_matrices(B21, B22))


    C11 = add_matrices(subtract_matrices(add_matrices(P5, P4), P2), P7)
    C12 = add_matrices(P1, P2)
    C21 = add_matrices(P3, P4)
    C22 = subtract_matrices(subtract_matrices(add_matrices(P5, P1), P3), P6)
    return join_matrices(C11, C12, C21, C22)

if __name__ == "__main__":
    print("Testing Strassen's Algorithm\n")
    print(f"Number of Iterations per test size: {iterations}")
    print("Matrix Size(N): Total Time (s): Average Time (ns):")
    for n in mat_size:
        A = createMatrix(n, random_fill=True)
        B = createMatrix(n, random_fill=True)
        start_time = time.time()
        for _ in range(iterations):
            C = strassens_Multiply(A, B)
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_ns = int(round((total_time / iterations) * 1e9))
        print(f"{n}: {total_time:.6f}: {avg_time_ns}")
    start_time = time.time()
    size=128
    A = createMatrix(size, random_fill=True)
    B = createMatrix(size, random_fill=True)
    C= strassens_Multiply(A, B)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Time taken to multiply two {size}x{size} matrices using Strassen\'s Algorithm: {total_time:.6f} seconds')
