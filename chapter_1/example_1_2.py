# Example 1.2: Simple linear regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
rng = np.random.default_rng(0)
# Hypothetical data: patient age (years) and systolic blood pressure (mmHg)
age = rng.uniform(20, 80, 60).reshape(-1, 1)
pressure = 90 + 0.7 * age.ravel() + rng.normal(0, 6, 60)
 
model = LinearRegression()
model.fit(age, pressure)
 
print("Slope (w):", round(model.coef_[0], 3))
print("Intercept (b):", round(model.intercept_, 3))
print("Predicted pressure at age 50:", round(model.predict([[50]])[0], 1))
print("MSE:", round(mean_squared_error(pressure, model.predict(age)), 2))
 
xs = np.linspace(20, 80, 100).reshape(-1, 1)
plt.figure(figsize=(6, 4.5))
plt.scatter(age, pressure, c="#3C6E8F", label="Patients")
plt.plot(xs, model.predict(xs), "r--", lw=2, label="Fitted line")
plt.xlabel("Age (years)"); plt.ylabel("Systolic pressure (mmHg)")
plt.legend(); plt.title("Linear regression")
plt.tight_layout(); plt.savefig("fig1_2.png", dpi=300)
