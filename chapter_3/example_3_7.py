# Example 3.7: Types of pooling — Max, Average, and Sum (the slide example)
import numpy as np
 
def pool2d(x, size=2, mode="max"):
    h, w = x.shape[0] // size, x.shape[1] // size
    blocks = x[:h*size, :w*size].reshape(h, size, w, size).swapaxes(1, 2)
    if mode == "max": return blocks.max(axis=(2, 3))
    if mode == "avg": return blocks.mean(axis=(2, 3))
    if mode == "sum": return blocks.sum(axis=(2, 3))
 
x = np.array([[ 2,  3, 2, 0],
              [ 5, -2, 2, 8],
              [-1, -6, 7, 3],
              [-4, -5, 4, 2]])
for mode in ["max", "avg", "sum"]:
    print(f"{mode.upper()} pooling:\n", pool2d(x, 2, mode))
