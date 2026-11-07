# Example 3.8: Tracing the dimensions and number of parameters of VGG-16 without any deep learning library
cfg = [64, 64, "P", 128, 128, "P", 256, 256, 256, "P",
       512, 512, 512, "P", 512, 512, 512, "P"]
 
size, ch, total = 224, 3, 0
print(f"{'layer':<14}{'output shape':<18}{'params':>12}")
for i, v in enumerate(cfg):
    if v == "P":
        size //= 2
        print(f"{'maxpool/2':<14}{f'{size}x{size}x{ch}':<18}{0:>12,}")
    else:
        params = 3 * 3 * ch * v + v          # weights + biases of a 3x3 conv
        ch, total = v, total + params
        print(f"{'conv3x3-' + str(v):<14}{f'{size}x{size}x{ch}':<18}{params:>12,}")
 
n_in = size * size * ch                       # flatten: 7*7*512
for units in (4096, 4096, 1000):
    params = n_in * units + units
    total += params
    print(f"{'fc-' + str(units):<14}{str(units):<18}{params:>12,}")
    n_in = units
print(f"Total parameters: {total:,}")
