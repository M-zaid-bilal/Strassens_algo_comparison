## How to Run

To reproduce the benchmarks:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Strassens-Performance.git
    cd Strassens-Performance
    ```
2.  **C++:**
    *   Navigate to the `cpp/` directory.
    *   Compile: `g++ Strassens_algorithm.cpp -o strassens_cpp -O3 -std=c++17`
    *   Run: `./strassens_cpp`
3.  **Python:**
    *   Navigate to the `python/` directory.
    *   Ensure you have Python installed (no `pip install` required for this phase).
    *   Run: `python Strassens_Algorithm_Matmult_Python.py`
4.  **R:**
    *   Navigate to the `r/` directory.
    *   Open an R console or terminal.
    *   Run: `source("Strassens_MatMult.R")` (or `Rscript Strassens_MatMult.R` from terminal)
5.  **Matlab:**
    *   Navigate to the `matlab/` directory in the Matlab environment.
    *   Run: Type `Strassens_Matmult_Matlab` in the Matlab Command Window and press Enter.

## Results (Phase 1: Pure Language Implementations)

The following tables and console outputs summarize the performance of Strassen's algorithm in each language under the "no external numerical libraries" constraint.

**Summary of Average Time (nanoseconds) per Iteration:**

| Matrix Size (N) | MATLAB (ns)   | R (ns)        | C++ (ns)       | Python (ns)      |
| :-------------- | :------------ | :------------ | :------------- | :--------------- |
| **2**           | 594,100       | 1,700,000     | 245,370        | 2,622,175        |
| **4**           | 447,200       | 400,000       | 2,110,360      | 14,576,054       |
| **8**           | 768,600       | 2,000,000     | 18,181,170     | 69,940,352       |
| **16**          | 3,499,000     | 4,100,000     | 98,052,710     | 529,642,820      |
| **32**          | 16,803,400    | 296,000,000   | 562,704,140    | 3,772,028,173    |
| **64**          | 114,371,300   | 2,051,000,000 | 4,878,091,730  | 26,513,939,977   |
| **128 (single)**| 816,692,000   | 14,500,000,000| 27,322,600,000 | 188,065,743,000  |

*(Note: "Total Time (s)" is for 10 iterations, "Average Time (nanoseconds)" is per single iteration, rounded to nearest whole number.)*

## Analysis and Discussion (Phase 1)

The results are highly illustrative of the nuances of "language performance" and the impact of underlying implementations:

1.  **MATLAB (Fastest):** MATLAB demonstrated the fastest performance by a significant margin. This is attributed to its highly optimized **JIT (Just-In-Time) compiler** and its core matrix operations, which are implemented in highly efficient C/Fortran code (often leveraging BLAS/LAPACK). Even though the Strassen's algorithm itself is written in high-level Matlab, the JIT efficiently optimizes the many recursive calls to `add_matrices`, `subtract_matrices`, `split_matrix`, and `join_matrices`, which internally use MATLAB's fast array primitives.

2.  **R (Second Fastest):** R follows MATLAB closely, also benefiting from its robust numerical foundation. Similar to MATLAB, R's `matrix` type and operators (`+`, `-`, indexing) are backed by highly optimized C/Fortran code (BLAS/LAPACK). This allows fundamental matrix operations to execute very quickly, despite R being an interpreted language. The slight difference compared to MATLAB could be due to differences in JIT effectiveness, memory management specifics for recursive calls, or function call overheads.

3.  **C++ (Third Fastest):** Our C++ implementation, using `std::vector<std::vector<double>>`, performed slower than both MATLAB and R. While C++ is a compiled language, this specific implementation suffers from:
    *   **Memory Fragmentation:** `std::vector<std::vector<double>>` leads to non-contiguous memory allocations for each row, resulting in more cache misses and less efficient data access compared to truly contiguous arrays.
    *   **Dynamic Allocation Overhead:** Every intermediate matrix (from additions, subtractions, splits, joins) requires dynamic memory allocation and deallocation, incurring significant overhead.
    *   **Manual Loop Overhead:** Our `addMatrices`, `subtractMatrices`, etc., are pure C++ loops, not optimized BLAS/LAPACK calls, which are much slower than the highly tuned routines used by R and MATLAB's core.

4.  **Python (Slowest):** The pure Python `list of lists` implementation was significantly the slowest. This is primarily due to:
    *   **Interpreted Language Overhead:** Python's dynamic typing and interpretation adds substantial overhead to every operation.
    *   **Object Overhead:** Each floating-point number in a Python `list of lists` is a full Python object, not a raw C `double`. This leads to massive memory usage and processing overhead (e.g., reference counting) for basic arithmetic.
    *   **Fragmented Memory:** Similar to `std::vector<std::vector<double>>`, `list of lists` is not contiguous in memory, further hindering performance.

### Key Takeaways from Phase 1:

*   **"No Libraries" is Context-Dependent:** The interpretation of "no external numerical libraries" varies drastically across languages. For R and MATLAB, it means using highly optimized built-in array primitives. For C++ and Python, it meant implementing these primitives from scratch, revealing significant performance penalties.
*   **The Power of Underlying C/Fortran Implementations:** Languages like R and MATLAB leverage decades of optimization in C/Fortran numerical libraries (BLAS/LAPACK) for their core matrix operations, leading to superior performance even for high-level algorithms written in their respective syntaxes.
*   **Data Structure Choice is Critical:** Even in C++, a compiled language, the choice of `std::vector<std::vector<double>>` over a contiguous 2D array or a dedicated numerical library has a profound impact on performance due to memory access patterns and allocation overhead.
*   **Python's Raw Numerical Performance is Low:** Without libraries like NumPy, Python is not suitable for performance-critical numerical computations due to its overhead.

## Future Work

This project serves as a foundational benchmark. The next steps will involve:

1.  **Optimized Implementations with Libraries (Phase 2):**
    *   **Python:** Implement Strassen's using NumPy (`np.ndarray`) for matrix operations.
    *   **C++:** Implement Strassen's using a dedicated numerical library like Eigen (`Eigen::MatrixXd`).
    *   **R & Matlab:** Their current implementations are already utilizing their optimized core matrix types. We will compare these against their built-in naive matrix multiplication functions (`%*%` in R, `*` in Matlab) which are also highly optimized.
    *   **Comparison:** Analyze the rate of improvement over the "pure language" implementations and compare the Strassen's algorithm against the library-optimized naive matrix multiplication in each language to identify crossover points.

2.  **AlphaTensor Integration (Phase 3):**
    *   Investigate and implement a published, specific AlphaTensor-derived matrix multiplication decomposition for small base cases, utilizing the optimized libraries from Phase 2.
    *   Compare the performance of this AlphaTensor-based approach against Strassen's and naive matrix multiplication.

3.  **Linear Algebra Practice:**
    *   Beyond matrix multiplication, implement fundamental linear algebra operations (e.g., matrix-vector products, transpose, determinant, inverse, eigenvalues using methods like power iteration) to deepen understanding of their computational aspects, especially for large matrices relevant to AI/ML.

This comprehensive approach will provide a holistic view of algorithm performance, language choice, and the impact of optimized libraries in real-world numerical computing scenarios, highly relevant for data science and machine learning applications.
