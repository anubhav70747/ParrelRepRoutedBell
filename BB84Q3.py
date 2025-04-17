import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import time  # Moved the import statement to the top
import matplotlib.pyplot as plt


def delta(a, b):
    """Kronecker delta function."""
    return int(a == b)

def A_matrix(x, a):
    """Compute the A[x, a] matrix."""
    if x == 0 and a == 0:
        return np.array([[1, 0], [0, 0]], dtype=np.float64)
    elif x == 0 and a == 1:
        return np.array([[0, 0], [0, 1]], dtype=np.float64)
    elif x == 1 and a == 0:
        return 0.5 * np.array([[1, 1], [1, 1]], dtype=np.float64)
    elif x == 1 and a == 1:
        return 0.5 * np.array([[1, -1], [-1, 1]], dtype=np.float64)

def A3_matrix(x0, x1, x2, a0, a1, a2, A_cache):
    """Compute the A3[x0, x1, x2, a0, a1, a2] matrix using Kronecker products."""
    return np.kron(A_cache[(x0, a0)], np.kron(A_cache[(x1, a1)], A_cache[(x2, a2)]))

def compute_CC(args):
    """Compute the maximum eigenvalue of CC for the given indices."""
    i, j, k, l, m, n, o, p, q, A_cache = args
    identity_matrix = np.identity(8, dtype=np.float64)
    CC = np.zeros((8, 8), dtype=np.float64)

    # Iterate over all combinations of a, b, c
    for a, b, c in product([0, 1], repeat=3):
        idx_abc = 4 * a + 2 * b + c  # Calculate idx_{abc}

        # Compute delta terms for each index
        delta_i = delta(i, idx_abc)
        delta_j = delta(j, idx_abc)
        delta_k = delta(k, idx_abc)
        delta_l = delta(l, idx_abc)
        delta_m = delta(m, idx_abc)
        delta_n = delta(n, idx_abc)
        delta_o = delta(o, idx_abc)
        delta_p = delta(p, idx_abc)

        # Check if any delta terms are non-zero
        if not any([delta_i, delta_j, delta_k, delta_l, delta_m, delta_n, delta_o, delta_p]):
            continue  # Skip to next a, b, c if all delta terms are zero

        # Accumulate the eight specific A3 terms
        CC += (
            delta_i * A3_matrix(0, 0, 0, a, b, c, A_cache) +
            delta_j * A3_matrix(0, 0, 1, a, b, c, A_cache) +
            delta_k * A3_matrix(0, 1, 0, a, b, c, A_cache) +
            delta_l * A3_matrix(0, 1, 1, a, b, c, A_cache) +
            delta_m * A3_matrix(1, 0, 0, a, b, c, A_cache) +
            delta_n * A3_matrix(1, 0, 1, a, b, c, A_cache) +
            delta_o * A3_matrix(1, 1, 0, a, b, c, A_cache) +
            delta_p * A3_matrix(1, 1, 1, a, b, c, A_cache)
        )

    # Apply the scaling factor of 1/64
    CC *= (1 / 64)

    # Compute delta_sum for the penalty term
    delta_sum = 8 - (
        delta(i, 8) + delta(j, 8) + delta(k, 8) + delta(l, 8) +
        delta(m, 8) + delta(n, 8) + delta(o, 8) + delta(p, 8)
    )

    # Apply the penalty term
    CC -= (q / 64) * delta_sum * identity_matrix

    # Compute eigenvalues and return the maximum eigenvalue
    eigenvalues = np.linalg.eigvalsh(CC)
    max_eigenvalue = np.max(eigenvalues)
    return max_eigenvalue

def generate_args(index_range, q, A_cache):
    """Generator that yields arguments for compute_CC."""
    for indices in product(index_range, repeat=8):
        yield (*indices, q, A_cache)

def BC(q):
    """Compute BC[q] as per the specified mathematical expression, keeping only the max eigenvalue."""
    # Precompute A[x, a] matrices and cache them
    A_cache = {(x, a): A_matrix(x, a) for x, a in product([0, 1], repeat=2)}

    index_range = range(8, -1, -1)  # Indices from 0 to 8 inclusive
    total_combinations = 9 ** 8  # Total number of combinations
    print(f"Total combinations to compute: {total_combinations}")

    # Use the generator to avoid loading all arguments into memory
    args_generator = generate_args(index_range, q, A_cache)

    max_eigenvalue = None
    processed = 0
    update_interval = 10000  # Update progress every 10,000 combinations
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(compute_CC, args_generator, chunksize=1000):
            if max_eigenvalue is None or result > max_eigenvalue:
                max_eigenvalue = result
            processed += 1
            if processed % update_interval == 0 or processed == total_combinations:
                elapsed_time = time.time() - start_time
                if processed > 0:
                    remaining_time = (elapsed_time / processed) * (total_combinations - processed)
                else:
                    remaining_time = 0
                current_BC_q = 8 * max_eigenvalue if max_eigenvalue is not None else 'N/A'
                print(f"Processed {processed}/{total_combinations} combinations. "
                      f"Elapsed time: {elapsed_time:.2f}s, "
                      f"Estimated remaining time: {remaining_time / 3600:.2f}h, "
                      f"Eta[q]: {current_BC_q/(1-q)}")

    return 8 * max_eigenvalue if max_eigenvalue is not None else None

# Example usage:
if __name__ == "__main__":
    # Define the range of q values from 0.47 to 0.48 with 10 steps
    q_values = np.linspace(0.7072, 0.7072, 1)
    results = []
    total_start_time = time.time()  # Track total computation time

    # Loop over the q values and compute BC for each
    for q_value in q_values:
        print(f"q = {q_value}")
        start_time = time.time()  # Time the computation for each q
        result = BC(q_value)/( (1 - q_value))
        end_time = time.time()

        # Store the result
        results.append(result)

        # Print the results for each q
        total_time = end_time - start_time
        print(f"eta[{q_value}] = {result}")
        print(f"Computation time for q = {q_value}: {total_time:.2f} seconds")

    total_end_time = time.time()
    total_computation_time = total_end_time - total_start_time
    print(f"Total computation time for all q values: {total_computation_time:.2f} seconds "
          f"({total_computation_time / 3600:.2f} hours)")

    # Plot q values against the corresponding BC results
    plt.figure(figsize=(8, 6))
    plt.plot(q_values, results, marker='o', linestyle='-', color='b')
    plt.xlabel('q')
    plt.ylabel('Crit det. efficiency')
    plt.grid(True)
    plt.show()