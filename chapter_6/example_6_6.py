# Example 6.6: A Mamdani fuzzy inference system from scratch — estimating a patient's risk level
# Inputs: body temperature (°C) and heart rate (bpm) -> Output: risk score 0..100
import numpy as np
import matplotlib.pyplot as plt
 
def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)
 
temp_mf = {"normal": lambda t: trimf(t, 35, 36.8, 38.0), "fever": lambda t: trimf(t, 37.2, 39, 41.5)}
hr_mf = {"normal": lambda h: trimf(h, 40, 75, 100), "high": lambda h: trimf(h, 85, 130, 180)}
risk = np.linspace(0, 100, 501)
risk_mf = {"low": trimf(risk, 0, 0, 45), "medium": trimf(risk, 25, 50, 75), "high": trimf(risk, 55, 100, 100)}
 
rules = [  # IF temp is A AND heart rate is B THEN risk is C
    ("normal", "normal", "low"),
    ("normal", "high", "medium"),
    ("fever", "normal", "medium"),
    ("fever", "high", "high"),
]
 
def infer(t, h, verbose=False):
    aggregated = np.zeros_like(risk)
    for tA, hB, out in rules:
        strength = min(temp_mf[tA](t), hr_mf[hB](h))                     # AND = min (fuzzification + rule firing)
        aggregated = np.maximum(aggregated, np.minimum(strength, risk_mf[out]))  # implication + aggregation
        if verbose and strength > 0:
            print(f"  rule: temp={tA:<6} AND hr={hB:<6} -> risk={out:<6} | firing strength = {strength:.2f}")
    if aggregated.sum() == 0: return 0.0, aggregated
    return (risk * aggregated).sum() / aggregated.sum(), aggregated     # centroid defuzzification
 
for t, h in [(36.8, 72), (37.8, 95), (39.5, 140)]:
    print(f"Patient: temp={t}°C, heart rate={h} bpm")
    score, agg = infer(t, h, verbose=True)
    print(f"  => crisp risk score = {score:.1f}\n")
 
_, agg = infer(37.8, 95)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for n, m in risk_mf.items(): ax[0].plot(risk, m, label=n)
ax[0].fill_between(risk, agg, alpha=0.4, color="#e9a03b", label="aggregated output")
ax[0].axvline(infer(37.8, 95)[0], c="k", ls="--"); ax[0].legend(fontsize=8); ax[0].set_title("Output for temp=37.8, HR=95")
T, H = np.meshgrid(np.linspace(35.5, 41, 45), np.linspace(50, 170, 45))
S = np.vectorize(lambda a, b: infer(a, b)[0])(T, H)
c = ax[1].contourf(T, H, S, levels=20, cmap="RdYlGn_r"); plt.colorbar(c, ax=ax[1])
ax[1].set_xlabel("temperature (°C)"); ax[1].set_ylabel("heart rate (bpm)"); ax[1].set_title("Control surface")
plt.tight_layout(); plt.savefig("fig6_4.png", dpi=300)
