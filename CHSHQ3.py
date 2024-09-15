import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import time  # Moved the import statement to the top

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

    # Sum over a, b, c in {0, 1}
    for a, b, c in product([0, 1], repeat=3):
        idx_abc = 4 * a + 2 * b + c  # Calculate idx_{abc}

        # Compute delta terms once per (a, b, c)
        delta_terms = [
            delta(i, idx_abc),
            delta(j, idx_abc),
            delta(k, idx_abc),
            delta(l, idx_abc),
            delta(m, idx_abc),
            delta(n, idx_abc),
            delta(o, idx_abc),
            delta(p, idx_abc)
        ]

        # Skip computation if all delta terms are zero
        if not any(delta_terms):
            continue

        # Sum over x0, x1, x2 in {0, 1}
        for x0, x1, x2 in product([0, 1], repeat=3):
            # Compute the eight terms as per your expression
            terms = [
                delta_terms[0] * A3_matrix(x0, x1, x2, a, b, c, A_cache),
                delta_terms[1] * A3_matrix(x0, x1, x2, a, b, c ^ x2, A_cache),
                delta_terms[2] * A3_matrix(x0, x1, x2, a, b ^ x1, c, A_cache),
                delta_terms[3] * A3_matrix(x0, x1, x2, a, b ^ x1, c ^ x2, A_cache),
                delta_terms[4] * A3_matrix(x0, x1, x2, a ^ x0, b, c, A_cache),
                delta_terms[5] * A3_matrix(x0, x1, x2, a ^ x0, b, c ^ x2, A_cache),
                delta_terms[6] * A3_matrix(x0, x1, x2, a ^ x0, b ^ x1, c, A_cache),
                delta_terms[7] * A3_matrix(x0, x1, x2, a ^ x0, b ^ x1, c ^ x2, A_cache),
            ]
            # Sum the terms directly into CC
            CC += sum(terms)

    # Multiply CC by the scalar factor
    CC *= (1 / 512)

    # Compute delta_sum directly using the delta function
    delta_sum = 8 - sum(delta(idx, 8) for idx in (i, j, k, l, m, n, o, p))
    # Subtract the q term
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

    index_range = range(9)  # Indices from 0 to 8 inclusive
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
                remaining_time = (elapsed_time / processed) * (total_combinations - processed)
                current_BC_q = 8 * max_eigenvalue if max_eigenvalue is not None else 'N/A'
                print(f"Processed {processed}/{total_combinations} combinations. "
                      f"Elapsed time: {elapsed_time:.2f}s, "
                      f"Estimated remaining time: {remaining_time / 3600:.2f}h, "
                      f"Current BC[q]: {current_BC_q}")

    return 8 * max_eigenvalue if max_eigenvalue is not None else None

# Example usage:
if __name__ == "__main__":
    import time  # Ensure time is imported
    q_value = 0.5  # Replace with your desired q value
    start_time = time.time()
    result = BC(q_value)
    end_time = time.time()
    print(f"BC[{q_value}] = {result}")
    total_time = end_time - start_time
    print(f"Total computation time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
