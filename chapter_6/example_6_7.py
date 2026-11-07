# Example 6.7: A Sugeno (TSK) fuzzy system — the output of each rule is a linear function
import numpy as np
 
def gauss(x, c, s): return np.exp(-((x - c) ** 2) / (2 * s ** 2))
 
# input: blood glucose (mg/dL) -> output: suggested insulin units (illustrative only!)
rules = [  # (membership center, width, linear consequent p*x + r)
    (100, 25, (0.00, 0.0)),     # IF glucose is normal THEN dose = 0
    (180, 35, (0.04, -4.0)),    # IF glucose is high   THEN dose = 0.04 x - 4
    (280, 45, (0.06, -6.0)),    # IF glucose is very high THEN dose = 0.06 x - 6
]
def sugeno(x):
    w = np.array([gauss(x, c, s) for c, s, _ in rules])
    f = np.array([p * x + r for _, _, (p, r) in rules])
    y = (w * f).sum() / w.sum()
    return max(y, 0.0), w / w.sum()          # a dose can not be negative
 
for g in (90, 150, 220, 300):
    y, wn = sugeno(g)
    print(f"glucose={g:>3} -> normalized weights {wn.round(2)} -> output = {y:.2f}")
