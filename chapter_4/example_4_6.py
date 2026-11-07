# Example 4.6: Upsampling — unpooling and transposed convolution
import numpy as np
 
x = np.array([[1, 2],
              [3, 4]])
 
nearest = x.repeat(2, axis=0).repeat(2, axis=1)              # nearest-neighbor upsampling
bed_of_nails = np.zeros((4, 4), int); bed_of_nails[::2, ::2] = x
 
def transpose_conv(x, k, stride=2):
    n, kk = x.shape[0], k.shape[0]
    out = np.zeros(((n - 1) * stride + kk,) * 2)
    for i in range(n):
        for j in range(n):          # every input pixel "stamps" a scaled copy of the kernel
            out[i*stride:i*stride+kk, j*stride:j*stride+kk] += x[i, j] * k
    return out
 
k = np.array([[1, 1, 0],
              [1, 1, 0],
              [0, 0, 0]])
print("Nearest-neighbor upsampling:\n", nearest)
print("Bed-of-nails unpooling:\n", bed_of_nails)
print("Transpose convolution (stride 2, 3x3 kernel):\n", transpose_conv(x, k).astype(int))
