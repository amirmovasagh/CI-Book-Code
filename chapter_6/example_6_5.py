# Example 6.5: Fuzzy operators — union, intersection, and complement
import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(35, 41, 601)                        # body temperature (°C)
normal = np.clip((38 - x) / 1.0, 0, 1) * (x >= 36) + (x < 36) * np.clip((x - 35) / 1.0, 0, 1)
fever = np.clip((x - 37) / 1.5, 0, 1)
 
union = np.maximum(normal, fever)                   # A ∪ B : max
inter = np.minimum(normal, fever)                   # A ∩ B : min
not_fever = 1 - fever                               # ¬B    : 1 - μ
prod = normal * fever                               # algebraic product (another t-norm)
 
t = 37.5; i = np.argmin(abs(x - t))
print(f"At {t}°C: μ_normal={normal[i]:.2f}, μ_fever={fever[i]:.2f}")
print(f"  union (max)={union[i]:.2f}, intersection (min)={inter[i]:.2f}, "
      f"product={prod[i]:.2f}, not fever={not_fever[i]:.2f}")
print("Law of excluded middle fails: max(μ, 1-μ) =", round(max(fever[i], 1 - fever[i]), 2))
 
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for a, (title, curve) in zip(ax, [("Union (max)", union), ("Intersection (min)", inter), ("Complement of fever", not_fever)]):
    a.plot(x, normal, "--", c="#4a90d9", label="normal"); a.plot(x, fever, "--", c="#d1495b", label="fever")
    a.fill_between(x, curve, alpha=0.35, color="#e9a03b"); a.set_title(title); a.set_xlabel("°C")
ax[0].legend(); plt.tight_layout(); plt.savefig("fig6_3.png", dpi=300)
