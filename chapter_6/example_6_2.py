# Example 6.2: Discrete fuzzy sets — each member with a degree of membership
# "Even days" of the Iranian week (Saturday, Monday, Wednesday); Friday is a holiday: partly even?
even_days = {"Saturday": 1.0, "Monday": 1.0, "Wednesday": 1.0, "Friday": 0.5}
# "Natural numbers close to 1"
near_one = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.1}
 
def mu(fuzzy_set, x):
    return fuzzy_set.get(x, 0.0)          # elements not listed have membership 0
 
print("μ_even(Monday)   =", mu(even_days, "Monday"))
print("μ_even(Friday)   =", mu(even_days, "Friday"))
print("μ_even(Tuesday)  =", mu(even_days, "Tuesday"))
print("μ_near_one(3)    =", mu(near_one, 3), "| μ_near_one(7) =", mu(near_one, 7))
 
support = [x for x, m in near_one.items() if m > 0]
core = [x for x, m in near_one.items() if m == 1]
print("support(near_one) =", support, "| core =", core, "| height =", max(near_one.values()))
for alpha in (0.2, 0.5):
    print(f"alpha-cut at {alpha}:", [x for x, m in near_one.items() if m >= alpha])
print("cardinality |near_one| =", sum(near_one.values()))
