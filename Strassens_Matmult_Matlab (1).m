
clear; 
clc;   

matrix_sizes = [2, 4, 8, 16, 32, 64]; 
num_iterations = 10;                
fprintf('Benchmarking Strassen''s Matrix Multiplication in Matlab\n');

fprintf('Number of Iterations per test size: %d\n', num_iterations);
fprintf('Matrix Sizes (N) : Total Time (s): Average Time (nanoseconds)\n');


total_duration_secs = zeros(1, length(matrix_sizes));

for k = 1:length(matrix_sizes)
    n = matrix_sizes(k);
    current_total_duration_sec = 0.0;

    for i = 1:num_iterations
        A = create_matrix(n, true); 
        B = create_matrix(n, true);
        
        tic; 
        C = strassens_multiply(A, B); 
        duration_sec = toc;
        
        current_total_duration_sec = current_total_duration_sec + duration_sec;

    end
    
    total_duration_secs(k) = current_total_duration_sec;
    average_duration_ns = (current_total_duration_sec / num_iterations) * 1e9;
    
    fprintf('%-15d : %.6f : %d\n', n, current_total_duration_sec, round(average_duration_ns));
end



n_large = 128;
A_large = create_matrix(n_large, true);
B_large = create_matrix(n_large, true);

tic; 
C_large = strassens_multiply(A_large, B_large);
time_taken_large = toc;


fprintf('Time taken (s) for %d by %d using Strassen''s: %.6f seconds\n', ...
        n_large, n_large, time_taken_large);


function mat = create_matrix(n, random_fill)
    if nargin < 2
        random_fill = false;
    end
    if random_fill
        mat = rand(n, n) * 10.0;
    else
        mat = zeros(n, n);
    end
end

function C = add_matrices(A, B)
    C = A + B;
end

function C = subtract_matrices(A, B) 
    C = A - B;
end

function sub_mat = split_matrix(P, r_start, c_start, size)
    sub_mat = P(r_start:(r_start + size - 1), c_start:(c_start + size - 1));
end

function C = join_matrices(C11, C12, C21, C22)
    half_n = size(C11, 1);
    n = half_n * 2;
    C = create_matrix(n);
    C(1:half_n, 1:half_n) = C11;
    C(1:half_n, (half_n + 1):n) = C12;
    C((half_n + 1):n, 1:half_n) = C21;
    C((half_n + 1):n, (half_n + 1):n) = C22;
end

function C = naive_matrix_multiply(A, B)
    C = A * B;
end

function C = strassens_multiply(A, B)
    n = size(A, 1);
    if n == 1
        C = [A(1, 1) * B(1, 1)];
        return;
    end
    half_n = n / 2;
    A11 = split_matrix(A, 1, 1, half_n);
    A12 = split_matrix(A, 1, half_n + 1, half_n);
    A21 = split_matrix(A, half_n + 1, 1, half_n);
    A22 = split_matrix(A, half_n + 1, half_n + 1, half_n);
    B11 = split_matrix(B, 1, 1, half_n);
    B12 = split_matrix(B, 1, half_n + 1, half_n);
    B21 = split_matrix(B, half_n + 1, 1, half_n);
    B22 = split_matrix(B, half_n + 1, half_n + 1, half_n);
    P1 = strassens_multiply(A11, subtract_matrices(B12, B22));
    P2 = strassens_multiply(add_matrices(A11, A12), B22);
    P3 = strassens_multiply(add_matrices(A21, A22), B11);
    P4 = strassens_multiply(A22, subtract_matrices(B21, B11));
    P5 = strassens_multiply(add_matrices(A11, A22), add_matrices(B11, B22));
    P6 = strassens_multiply(subtract_matrices(A21, A11), add_matrices(B11, B12));
    P7 = strassens_multiply(subtract_matrices(A12, A22), add_matrices(B21, B22));
    C11 = add_matrices(subtract_matrices(add_matrices(P5, P4), P2), P7);
    C12 = add_matrices(P1, P2);
    C21 = add_matrices(P3, P4);
    C22 = subtract_matrices(subtract_matrices(add_matrices(P5, P1), P3), P6);
    C = join_matrices(C11, C12, C21, C22);
end