# Example 3.5: Convolution with a sharpening kernel and zero padding (the slide example)
import numpy as np
from scipy.signal import correlate2d
 
img = np.array([[ 60, 113,  56, 139,  85],
                [ 73, 121,  54,  84, 128],
                [131,  99,  70, 129, 127],
                [ 80,  57, 115,  69, 134],
                [104, 126, 123,  95, 130]])
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])
 
# mode="same" + fill=0  == zero padding of 1 pixel -> output is also 5x5
out = correlate2d(img, kernel, mode="same", boundary="fill", fillvalue=0)
print("Output feature map (5x5):\n", out)
