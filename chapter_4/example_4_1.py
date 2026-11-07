# Example 4.1: Computing the feature map size with the formula L' = (L - K + 2P) / S + 1
def feature_map_size(L, K, P=0, S=1):
    return (L - K + 2 * P) // S + 1
 
# three examples from the slides: smaller, same and larger output
for L, K, P, S in [(5, 3, 1, 2), (5, 3, 1, 1), (5, 3, 2, 1)]:
    print(f"L={L}, K={K}, P={P}, S={S}  ->  L' = {feature_map_size(L, K, P, S)}")
 
# "same" padding for an odd kernel: P = (K - 1) / 2 with S = 1
for K in (3, 5, 7):
    print(f"K={K}: same padding P={(K - 1) // 2} keeps 224 -> {feature_map_size(224, K, (K - 1) // 2)}")
