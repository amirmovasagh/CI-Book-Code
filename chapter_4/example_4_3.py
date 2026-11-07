# Example 4.3: Reconstructing the CNN architecture table of the glioblastoma classification paper and computing dimensions and parameters
def out_size(L, K, P, S): return (L - K + 2 * P) // S + 1
 
layers = [("Conv1+ReLU", "conv", 5, 32), ("Pool1", "pool", 2, None),
          ("Conv2+ReLU", "conv", 5, 32), ("Pool2", "pool", 2, None),
          ("Conv3+ReLU", "conv", 5, 48), ("Pool3", "pool", 2, None),
          ("Conv4+ReLU", "conv", 5, 48), ("Pool4", "pool", 2, None),
          ("Conv5+ReLU", "conv", 5, 64), ("Conv6+ReLU", "conv", 5, 64),
          ("Pool6", "pool", 2, None), ("Conv7+ReLU", "conv", 5, 128)]
 
size, ch, total = 224, 3, 0          # input image resized from 1000x1000 to 224x224 (RGB)
print(f"{'layer':<12}{'filter':<8}{'maps':>6}{'spatial':>12}{'params':>12}")
for name, kind, k, n in layers:
    if kind == "conv":
        size = out_size(size, k, (k - 1) // 2, 1)          # 'same' padding, stride 1
        params = k * k * ch * n + n; ch = n
    else:
        size = out_size(size, k, 0, 2); params = 0
    total += params
    print(f"{name:<12}{f'{k}x{k}':<8}{ch:>6}{f'{size}x{size}':>12}{params:>12,}")
flat = size * size * ch
for name, units in [("FC1+ReLU", 1024), ("FC2+Softmax", 2)]:
    params = flat * units + units; total += params; flat = units
    print(f"{name:<12}{'':<8}{units:>6}{'1x1':>12}{params:>12,}")
print(f"Total trainable parameters: {total:,}")
