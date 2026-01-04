matrix_sizes <- c(2, 4, 8, 16, 32, 64)
num_iterations <- 10

create_matrix <- function(n, random_fill = FALSE) {
  if (random_fill) {
    mat <- matrix(runif(n * n, min = 0.0, max = 10.0), nrow = n, ncol = n)
  } else {
    mat <- matrix(0.0, nrow = n, ncol = n)
  }
  return(mat)
}

add_matrices <- function(A, B) {
  return(A + B)
}

subtract_matrices <- function(A, B) {
  return(A - B)
}

split_matrix <- function(P, r_start, c_start, size) {
  sub_mat <- P[r_start:(r_start + size - 1), c_start:(c_start + size - 1)]
  return(sub_mat)
}

join_matrices <- function(C11, C12, C21, C22) {
  half_n <- nrow(C11) 
  n <- half_n * 2     
  C <- create_matrix(n) 
  
  C[1:half_n, 1:half_n] <- C11                   
  C[1:half_n, (half_n + 1):n] <- C12             
  C[(half_n + 1):n, 1:half_n] <- C21             
  C[(half_n + 1):n, (half_n + 1):n] <- C22      
  
  return(C)
}

naive_matrix_multiply <- function(A, B) {
  return(A %*% B)
}
strassens_multiply <- function(A, B) {
  n <- nrow(A)
  
  # Base case: 1x1
  if (n == 1) {
    return(matrix(A[1, 1] * B[1, 1], 1, 1))
  }
  
  if (n %% 2 != 0) {
    stop("Matrix size must be even")
  }
  
  half_n <- n / 2
  
  A11 <- A[1:half_n, 1:half_n, drop = FALSE]
  A12 <- A[1:half_n, (half_n+1):n, drop = FALSE]
  A21 <- A[(half_n+1):n, 1:half_n, drop = FALSE]
  A22 <- A[(half_n+1):n, (half_n+1):n, drop = FALSE]
  
  B11 <- B[1:half_n, 1:half_n, drop = FALSE]
  B12 <- B[1:half_n, (half_n+1):n, drop = FALSE]
  B21 <- B[(half_n+1):n, 1:half_n, drop = FALSE]
  B22 <- B[(half_n+1):n, (half_n+1):n, drop = FALSE]
  
  P1 <- strassens_multiply(A11, B12 - B22)
  P2 <- strassens_multiply(A11 + A12, B22)
  P3 <- strassens_multiply(A21 + A22, B11)
  P4 <- strassens_multiply(A22, B21 - B11)
  P5 <- strassens_multiply(A11 + A22, B11 + B22)
  P6 <- strassens_multiply(A21 - A11, B11 + B12)
  P7 <- strassens_multiply(A12 - A22, B21 + B22)
  
  C11 <- P5 + P4 - P2 + P7
  C12 <- P1 + P2
  C21 <- P3 + P4
  C22 <- P5 + P1 - P3 - P6
  
  rbind(
    cbind(C11, C12),
    cbind(C21, C22)
  )
}



message("Benchmarking Strassen's Matrix Multiplication in R")

message(paste("Number of Iterations per test size:", num_iterations))
message("Matrix Sizes (N) : Total Time (s): Average Time (nanoseconds)")


for (n in matrix_sizes) {
  total_duration_sec <- 0.0 
  
  for (i in 1:num_iterations) {
    A <- create_matrix(n, random_fill = TRUE)
    B <- create_matrix(n, random_fill = TRUE)
    
    time_taken <- system.time({
      C <- strassens_multiply(A, B)
    })
    
    total_duration_sec <- total_duration_sec + time_taken["elapsed"]
  }
  
  average_duration_ns <- (total_duration_sec / num_iterations) * 1e9
  message(sprintf("%-15d : %.6f : %d", n, total_duration_sec, round(average_duration_ns)))
}


n_large <- 128
A_large <- create_matrix(n_large, random_fill = TRUE)
B_large <- create_matrix(n_large, random_fill = TRUE)

time_taken_large <- system.time({
  C_large <- strassens_multiply(A_large, B_large)
})

message(sprintf("Time taken (s) for %d by %d using Strassen's: %.6f seconds", 
                n_large, n_large, time_taken_large["elapsed"]))
