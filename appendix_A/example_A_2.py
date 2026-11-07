# Example A.2: Data structures — lists, tuples, dictionaries, and sets
heart_rates = [72, 88, 95, 61, 110]           # list (mutable)
heart_rates.append(79)
print("List:", heart_rates, "| length:", len(heart_rates), "| max:", max(heart_rates))
print("Slicing first three:", heart_rates[:3], "| last:", heart_rates[-1])
 
shape = (512, 512, 3)                         # tuple (immutable)
print("Tuple:", shape, "| height:", shape[0])
 
patient = {"id": 17, "age": 52, "diagnosis": "benign"}   # dictionary
patient["follow_up"] = True
print("Dict keys:", list(patient.keys()), "| diagnosis:", patient["diagnosis"])
 
blood_types_a = {"A", "B", "O"}; blood_types_b = {"O", "AB"}  # sets
print("Union:", sorted(blood_types_a | blood_types_b), "| intersection:", blood_types_a & blood_types_b)
 
squares = [x ** 2 for x in range(6)]          # list comprehension
print("List comprehension:", squares)
