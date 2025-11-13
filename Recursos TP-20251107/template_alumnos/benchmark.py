import numpy as np
from alc import *
import time

# Define the number of matrices in the series
num_matrices_in_series = 500

# Define the starting number of rows
start_rows = 100

print(f"Generating a series of {num_matrices_in_series} matrices where shape = (N, N+1)\n")

print("Starting SVD timing test...\n")

# A list to store our results
timing_results = []

for i in range(num_matrices_in_series):
    
    # 1. Create the progressively larger matrix
    num_rows = start_rows + i
    num_cols = num_rows + 1
    random_matrix = np.random.rand(num_rows, num_cols)
    
    print(f"Iteration {i+1}: Matrix Shape {random_matrix.shape}")

    # 2. Start the timer
    start_time = time.perf_counter()
    
    # 3. Run the specific function you want to measure
    svd_reducida(random_matrix)
    
    # 4. Stop the timer
    end_time = time.perf_counter()
    
    # 5. Calculate and store the duration
    duration = end_time - start_time
    timing_results.append((random_matrix.shape, duration))
    
    print(f"  Time taken: {duration:.8f} seconds\n")

# ---
# 6. Print a summary at the end
# ---
print("-" * 40)
print("Timing Summary:")
print("Shape      | Time (seconds)")
print("-----------|---------------")
for shape, duration in timing_results:
    print(f"{str(shape):<10} | {duration:.8f}")