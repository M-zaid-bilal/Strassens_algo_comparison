#include <iostream>   
#include <vector>     // Not used for matrices, but included in original.
#include <chrono>    
#include <random>     
#include <iomanip>    

#include <eigen-5.0.0/Eigen/dense>


using namespace std;


using Matrix = Eigen::MatrixXd; 


class Stopwatch {
public:
    Stopwatch()
        : _start_time(std::chrono::high_resolution_clock::now()), _stopped(false) {
    }

    void stop() {
        if (!_stopped) {
            _end_time = std::chrono::high_resolution_clock::now();
            _stopped = true;
        }
    }

    void start() {
        _start_time = std::chrono::high_resolution_clock::now();
        _stopped = false;
    }

    long long get_duration_ns() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - _start_time).count();
    }

    long long get_duration_us() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - _start_time).count();
    }

    long long get_duration_ms() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _start_time).count();
    }

    double get_duration_s() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - _start_time;
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point _start_time;
    std::chrono::high_resolution_clock::time_point _end_time;
    bool _stopped;
};

    const std::vector<int> MATRIX_SIZES = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,2048,4096,8192,16384 };
    const int NUM_ITERATIONS = 10;
    const int MAX_STRASSEN_SIZE = 64;

    Matrix create_matrix_eigen(int n, bool random_fill = false) {
        Matrix mat(n, n); // Eigen matrix of size n x n

        if (random_fill) {
            
            mat = Matrix::Random(n, n) * 5.0 + Matrix::Constant(n, n, 5.0);
        }
        else {
            mat = Matrix::Zero(n, n); 
        }
    return mat;
    }

    Matrix add_matrices_eigen(const Matrix& A, const Matrix& B) {
        return A + B; 
    }

    Matrix subtract_matrices_eigen(const Matrix& A, const Matrix& B) {
        return A - B; 
	}
    Matrix multiply_matrices_eigen(const Matrix& A, const Matrix& B) {
        return A * B; 
	}
    Matrix split_matrix_eigen(const Matrix& P, int r_start, int c_start, int size) {
        return P.block(r_start, c_start, size, size);
    }
    Matrix join_matrices_eigen(const Matrix& C11, const Matrix& C12, const Matrix& C21, const Matrix& C22) {
        int half_n = C11.rows();
        int n = half_n * 2;
        Matrix C(n, n); 
        C.block(0, 0, half_n, half_n) = C11;
        C.block(0, half_n, half_n, half_n) = C12;
        C.block(half_n, 0, half_n, half_n) = C21;
        C.block(half_n, half_n, half_n, half_n) = C22;

        return C;
    }

    Matrix naive_matrix_multiply_eigen(const Matrix& A, const Matrix& B) {
        return A * B;
    }
    Matrix strassens_multiply_eigen(const Matrix& A, const Matrix& B) {
        int n = A.rows(); // Get dimension

        // Base case: If matrix size is 1x1, perform scalar multiplication
        if (n == 1) {
            Matrix C(1, 1);
            C(0, 0) = A(0, 0) * B(0, 0); // Access elements using (row, col)
            return C;
        }

        int half_n = n / 2;

        // Divide matrices A and B into 4 sub-matrices
        Matrix A11 = split_matrix_eigen(A, 0, 0, half_n);
        Matrix A12 = split_matrix_eigen(A, 0, half_n, half_n);
        Matrix A21 = split_matrix_eigen(A, half_n, 0, half_n);
        Matrix A22 = split_matrix_eigen(A, half_n, half_n, half_n);

        Matrix B11 = split_matrix_eigen(B, 0, 0, half_n);
        Matrix B12 = split_matrix_eigen(B, 0, half_n, half_n);
        Matrix B21 = split_matrix_eigen(B, half_n, 0, half_n);
        Matrix B22 = split_matrix_eigen(B, half_n, half_n, half_n);

        // Core Strassen's Algo: P1 TO P7 (recursive calls)
        Matrix P1 = strassens_multiply_eigen(A11, subtract_matrices_eigen(B12, B22));
        Matrix P2 = strassens_multiply_eigen(add_matrices_eigen(A11, A12), B22);
        Matrix P3 = strassens_multiply_eigen(add_matrices_eigen(A21, A22), B11);
        Matrix P4 = strassens_multiply_eigen(A22, subtract_matrices_eigen(B21, B11));
        Matrix P5 = strassens_multiply_eigen(add_matrices_eigen(A11, A22), add_matrices_eigen(B11, B22));
        Matrix P6 = strassens_multiply_eigen(subtract_matrices_eigen(A21, A11), add_matrices_eigen(B11, B12));
        Matrix P7 = strassens_multiply_eigen(subtract_matrices_eigen(A12, A22), add_matrices_eigen(B21, B22));

       
        Matrix C11_temp1 = add_matrices_eigen(P5, P4);
        Matrix C11_temp2 = subtract_matrices_eigen(C11_temp1, P2);
        Matrix C11 = add_matrices_eigen(C11_temp2, P7);
        Matrix C12 = add_matrices_eigen(P1, P2);
        Matrix C21 = add_matrices_eigen(P3, P4);
        Matrix C22_temp1 = subtract_matrices_eigen(P1, P3);
        Matrix C22_temp2 = subtract_matrices_eigen(P5, P6);
        Matrix C22 = add_matrices_eigen(C22_temp1, C22_temp2);

        return join_matrices_eigen(C11, C12, C21, C22);
    }

    int main() {
        cout << "Benchmarking Strassen's and Naive Matrix Multiplication in C++ with Eigen" << endl;
        cout << "Number of Iterations per test size: " << NUM_ITERATIONS << endl;
        cout << "Matrix Size (N) | Total Time Naive (s) | Avg Time Naive (ns) | Total Time Strassen (s) | Avg Time Strassen (ns)" << endl;
        

        for (int n : MATRIX_SIZES) {
            double total_time_naive_s = 0.0;
            double total_time_strassens_s = 0.0;
            bool run_strassen = (n <= MAX_STRASSEN_SIZE);

            for (int it = 0; it < NUM_ITERATIONS; ++it) {
                Matrix A = create_matrix_eigen(n, true);
                Matrix B = create_matrix_eigen(n, true);
				//Bemchmark Naive Multiplication
                Stopwatch sw_naive;
                Matrix C_naive = naive_matrix_multiply_eigen(A, B);
                sw_naive.stop();
                total_time_naive_s += sw_naive.get_duration_s();
                volatile double dummy_sum_naive = C_naive.sum(); // Prevent optimization
				//Benchmark Strassen's Multiplication (only up to MAX_STRASSEN_SIZE)
                if (run_strassen) {
                    Stopwatch sw_strassens;
                    Matrix C_strassens = strassens_multiply_eigen(A, B);
                    sw_strassens.stop();
                    total_time_strassens_s += sw_strassens.get_duration_s();
                    volatile double dummy_sum_strassens = C_strassens.sum(); // Prevent optimization
                }
            }

            double avg_time_naive_ns = (total_time_naive_s * 1e9) / NUM_ITERATIONS;

            if (run_strassen) {
                double avg_time_strassens_ns = (total_time_strassens_s * 1e9) / NUM_ITERATIONS;
                cout << setw(15) << n
                    << " : " << setw(20) << fixed << setprecision(6) << total_time_naive_s
                    << " : " << setw(18) << fixed << setprecision(0) << avg_time_naive_ns
                    << " : " << setw(23) << fixed << setprecision(6) << total_time_strassens_s
                    << " : " << setw(21) << fixed << setprecision(0) << avg_time_strassens_ns << endl;
            } else {
                cout << setw(15) << n
                    << " : " << setw(20) << fixed << setprecision(6) << total_time_naive_s
                    << " : " << setw(18) << fixed << setprecision(0) << avg_time_naive_ns
                    << " : " << setw(23) << "N/A"
                    << " : " << setw(21) << "N/A" << endl;
            }
        }
        // --- Single run for largest matrix size (1024x1024) for confirmation ---
        int n_large = 1024;
        Matrix A_large = create_matrix_eigen(n_large, true);
        Matrix B_large = create_matrix_eigen(n_large, true);

        cout << "Single run for " << n_large << "x" << n_large << " (1 iteration):" << endl;

        Stopwatch sw_naive_large;
        Matrix C_naive_large = naive_matrix_multiply_eigen(A_large, B_large);
        sw_naive_large.stop();
        volatile double dummy_sum_naive_large = C_naive_large.sum();
        cout << "Naive Eigen: " << fixed << setprecision(6) << sw_naive_large.get_duration_s() << " seconds" << endl;

        if (n_large <= MAX_STRASSEN_SIZE) {
            Stopwatch sw_strassens_large;
            Matrix C_strassens_large = strassens_multiply_eigen(A_large, B_large);
            sw_strassens_large.stop();
            volatile double dummy_sum_strassens_large = C_strassens_large.sum();
            cout << "Strassen's Eigen: " << fixed << setprecision(6) << sw_strassens_large.get_duration_s() << " seconds" << endl;
        } else {
            cout << "Strassen's Eigen: N/A (only supported up to " << MAX_STRASSEN_SIZE << "x" << MAX_STRASSEN_SIZE << ")" << endl;
        }
        cout << "Note: Results may vary based on system, compiler, Eigen version, and underlying BLAS/LAPACK libraries (e.g., MKL, OpenBLAS)." << endl;

        return 0;
    }