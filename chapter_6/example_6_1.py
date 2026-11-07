# Example 6.1: Classical (crisp) sets and the characteristic function in Python
A = {1, 2, 3, 4, 5}
 
def indicator(x, S):
    """I_A(x) = 1 if x ∈ A else 0"""
    return 1 if x in S else 0
 
for x in (1, 2, 6):
    print(f"{x} ∈ A ? {x in A}  ->  I_A({x}) = {indicator(x, A)}")
 
vowels = {"a", "e", "i", "o", "u"}
print("'y' ∈ V ?", "y" in vowels, " <- classical logic forces a yes/no answer")
