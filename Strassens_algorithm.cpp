// Strassens_algorithm.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>

using namespace std;
using Matrix = vector<vector<double>>;
class Stopwatch {
public:
    Stopwatch()
        : _start_time(std::chrono::high_resolution_clock::now()), _stopped(false) {
    }

    void stop() {
        // Record end time once; idempotent.
        if (!_stopped) {
            _end_time = std::chrono::high_resolution_clock::now();
            _stopped = true;
        }
    }

    void start() {
        _start_time = std::chrono::high_resolution_clock::now();
        _stopped = false;
    }

    // Returns duration in nanoseconds
    long long get_duration_ns() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - _start_time).count();
    }

    // Returns duration in microseconds
    long long get_duration_us() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - _start_time).count();
    }

    // Returns duration in milliseconds (for longer operations)
    long long get_duration_ms() const {
        auto end_time = _stopped ? _end_time : std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _start_time).count();
    }

    // Returns duration in seconds (double, for convenience)
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
Matrix create_matrix(int n, bool random_fill = false) {
    Matrix mat(n, vector<double>(n));
	if (random_fill) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		static std::uniform_real_distribution<> dis(0.0, 10.0);
        for(int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                mat[i][j] = dis(gen);
            }
		}
        }
	return mat;
}
Matrix printMatrix(const Matrix& mat) {
    int n = mat.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; j++) {
            cout << fixed << setprecision(2) << mat[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
	return mat;
}
Matrix addMatrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<double>(n));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Matrix subtractMatrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<double>(n));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}


Matrix splitMatrix(const Matrix& P, int r_start, int c_start, int size) {
	Matrix sub_matrix = create_matrix(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            sub_matrix[i][j] = P[r_start + i][c_start + j];
        }
	}
	return sub_matrix;
}
Matrix joinMatrices(const Matrix& C11, const Matrix& C12, const Matrix& C21, const Matrix& C22) {
    int half_n = C11.size();
    int n = half_n * 2;
    Matrix C = create_matrix(n);
    for (int i = 0; i < half_n; ++i) {
        for (int j = 0; j < half_n; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + half_n] = C12[i][j];
            C[i + half_n][j] = C21[i][j];
            C[i + half_n][j + half_n] = C22[i][j];

        }
    }
    return C;
}

Matrix naiveMatMult(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C = create_matrix(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix Strassensmatmult(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n == 1) {
        Matrix C = create_matrix(1);
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }
    int half_n = n / 2;
    Matrix A11 = splitMatrix(A, 0, 0, half_n);
    Matrix A12 = splitMatrix(A, 0, half_n, half_n);
    Matrix A21 = splitMatrix(A, half_n, 0, half_n);
    Matrix A22 = splitMatrix(A, half_n, half_n, half_n);
    Matrix B11 = splitMatrix(B, 0, 0, half_n);
    Matrix B12 = splitMatrix(B, 0, half_n, half_n);
    Matrix B21 = splitMatrix(B, half_n, 0, half_n);
    Matrix B22 = splitMatrix(B, half_n, half_n, half_n);
    //Core Strassen's Algo the P1 TO P7 
    /* Recall for Strassen's Algo:
    P1= A11(B12- B22)
    P2= (A11+A12)(B22)
    P3= (A21+A22)(B11)
    P4= A22(B21-B11)
    P5= (A11+A22)(B11+B22)
    P6= (A21-A11)(B11+B12)
    P7= (A12-A22)(B21+B22)
    
    */

    Matrix P1 = Strassensmatmult(A11, subtractMatrices(B12, B22));
    Matrix P2 = Strassensmatmult(addMatrices(A11, A12), B22);
    Matrix P3 = Strassensmatmult(addMatrices(A21, A22), B11);
    Matrix P4 = Strassensmatmult(A22, subtractMatrices(B21, B11));
    Matrix P5 = Strassensmatmult(addMatrices(A11, A22), addMatrices(B11, B22));
    Matrix P6 = Strassensmatmult(subtractMatrices(A21, A11), addMatrices(B11, B12));
    Matrix P7 = Strassensmatmult(subtractMatrices(A12, A22), addMatrices(B21, B22));

    /*
    The Products are:
    C11= P5+P4-P2+P7
    C12= P1+P2
    C21= P3+P4
    C22= P5+P1-P3-P6
    */

    Matrix C11_temp1 = addMatrices(P5, P4);
    Matrix C11_temp2 = subtractMatrices(C11_temp1, P2);
    Matrix C11 = addMatrices(C11_temp2, P7);
    Matrix C12 = addMatrices(P1, P2);
    Matrix C21 = addMatrices(P3, P4);
    Matrix C22_temp1 = subtractMatrices(P1, P3);
    Matrix C22_temp2 = subtractMatrices(P5, P6);
    Matrix C22 = addMatrices(C22_temp1, C22_temp2);
    
    return joinMatrices(C11, C12, C21, C22);
}

int main() {
    vector<int> testsizes = { 2,4,8,16,32,64};
    const int iterations = 10;
	cout << "Number of Iterations per test size: " << iterations << endl;
    cout << "Matrix Sizes (N) : Total Time (s): Average Time (nanoseconds) " << endl;

    for (int n : testsizes) {
        int iter = iterations;
        // prepare random input matrices
        Matrix A = create_matrix(n, true);
        Matrix B = create_matrix(n, true);
        Strassensmatmult(A, B);

        Stopwatch sw;
        for (int it = 0; it < iter; ++it) {
            Matrix C = Strassensmatmult(A, B);
        }
        sw.stop();
        double total_time_s = sw.get_duration_s();
        double avg_time_ns = (total_time_s * 1e9) / iter;
        cout << setw(15) << n << " : " << setw(14) << fixed << setprecision(6) << total_time_s
            << " : " << setw(24) << fixed << setprecision(0) << avg_time_ns << endl;
		
		
    }
	
	int test_size = 128;
	Matrix A = create_matrix(test_size, true);
	Matrix B = create_matrix(test_size, true);
    Stopwatch sw2;
    sw2.start();
	Matrix C_strassen = Strassensmatmult(A, B);
    sw2.stop();
	cout << "Time taken (s) for 128 by 128 using Strassen's: " << fixed << setprecision(6) << sw2.get_duration_s() << " seconds" << endl;

    return 0;
}