# Example 3.4: Manual convolution on a 5×5 binary image (the slide example)
import numpy as np
 
def conv2d(img, kernel, stride=1, padding=0):
    if padding:
        img = np.pad(img, padding)                     # zero padding
    k = kernel.shape[0]
    out = (img.shape[0] - k) // stride + 1             # output size
    fmap = np.zeros((out, out))
    for i in range(out):
        for j in range(out):
            patch = img[i*stride:i*stride+k, j*stride:j*stride+k]
            fmap[i, j] = np.sum(patch * kernel)        # element-wise product, then sum
    return fmap
 
image = np.array([[1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]])
kernel = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
print("Convolved feature:\n", conv2d(image, kernel).astype(int))
