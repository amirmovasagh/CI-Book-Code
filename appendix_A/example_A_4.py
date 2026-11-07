# Example A.4: NumPy arrays and vectorized computation
import numpy as np
 
a = np.array([1.0, 2.0, 3.0, 4.0])
M = np.arange(12).reshape(3, 4)
print("a * 2 =", a * 2, "| a ** 2 =", a ** 2)
print("M shape:", M.shape, "\nM =\n", M)
print("Column means:", M.mean(axis=0), "| row sums:", M.sum(axis=1))
print("Boolean mask M > 5:\n", M > 5)
print("Elements greater than 5:", M[M > 5])
print("Matrix product M @ M.T:\n", M @ M.T)
 
rng = np.random.default_rng(0)
x = rng.normal(loc=100, scale=15, size=10000)    # e.g. simulated IQ / lab values
print(f"Sample mean={x.mean():.2f}, std={x.std():.2f}, min={x.min():.1f}, max={x.max():.1f}")
print("Broadcasting (3x1) + (1x4):\n", np.array([[0], [10], [20]]) + np.array([1, 2, 3, 4]))
