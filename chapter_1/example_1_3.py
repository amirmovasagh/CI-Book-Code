# Example 1.3: One problem, two views — regression versus classification
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
 
rng = np.random.default_rng(1)
today = rng.uniform(0, 40, 300)                     # today's temperature (°C)
tomorrow = 0.9 * today + 2 + rng.normal(0, 2, 300)  # tomorrow's temperature (°C)
X = today.reshape(-1, 1)
 
# View 1: regression — "What will tomorrow's temperature be?"
reg = LinearRegression().fit(X, tomorrow)
print("Regression  -> tomorrow at 28°C today:",
      round(reg.predict([[28]])[0], 1), "°C")
 
# View 2: classification — "Will tomorrow be hot or cold?" (threshold: 25 degrees)
labels = (tomorrow > 25).astype(int)               # 1 = Hot, 0 = Cold
clf = LogisticRegression().fit(X, labels)
pred = clf.predict([[28]])[0]
print("Classification -> tomorrow at 28°C today:",
      "HOT" if pred == 1 else "COLD",
      "| P(hot) =", round(clf.predict_proba([[28]])[0, 1], 2))
