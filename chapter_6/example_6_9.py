# Example 6.9: Fuzzy relations and max-min composition — medical diagnosis using Sanchez's method
import numpy as np
 
symptoms = ["fever", "cough", "headache", "chest pain"]
diseases = ["common cold", "influenza", "pneumonia"]
# R: how strongly each symptom is associated with each disease (expert knowledge, illustrative)
R = np.array([[0.4, 0.9, 0.8],     # fever
              [0.8, 0.7, 0.9],     # cough
              [0.5, 0.8, 0.3],     # headache
              [0.1, 0.2, 0.9]])    # chest pain
 
patients = {"patient A": [0.3, 0.9, 0.4, 0.0], "patient B": [0.9, 0.6, 0.9, 0.1], "patient C": [0.8, 0.9, 0.2, 0.8]}
for name, s in patients.items():
    s = np.array(s)
    d = np.max(np.minimum(s[:, None], R), axis=0)        # max-min composition:  D = S o R
    best = diseases[int(np.argmax(d))]
    print(f"{name}: symptoms={s} -> disease memberships {', '.join(f'{n}={v:.1f}' for n, v in zip(diseases, d))} -> {best}")
