# Example A.1: Python basics — variables, data types, and operators
age = 52                      # int
temperature = 37.8            # float
name = "Patient-017"          # str
has_fever = temperature >= 37.5   # bool
 
print(type(age), type(temperature), type(name), type(has_fever))
print(f"{name}: age={age}, temp={temperature}°C, fever={has_fever}")
print("BMI example:", round(82 / 1.75 ** 2, 1))
print("Integer division and remainder:", 17 // 5, 17 % 5)
