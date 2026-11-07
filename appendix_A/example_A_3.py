# Example A.3: Control structures and functions
def classify_bp(systolic, diastolic):
    """Classify blood pressure (simplified categories, for teaching only)."""
    if systolic < 120 and diastolic < 80:
        return "normal"
    elif systolic < 130 and diastolic < 80:
        return "elevated"
    elif systolic < 140 or diastolic < 90:
        return "stage 1"
    else:
        return "stage 2"
 
readings = [(115, 75), (125, 78), (135, 85), (150, 95)]
for s, d in readings:
    print(f"{s}/{d} mmHg -> {classify_bp(s, d)}")
 
count = 0
while count < 3:
    count += 1
print("while loop finished after", count, "iterations")
 
mean = lambda values: sum(values) / len(values)   # anonymous function
print("Mean systolic:", mean([s for s, _ in readings]))
