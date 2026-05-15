import pandas as pd
from scipy.stats import mannwhitneyu

# Example structure of your dataset
# Each row = one function extracted from a PR
data = pd.DataFrame({
    "source": ["AI", "AI", "AI", "AI", "Human", "Human", "Human", "Human"],
    "ccn": [2.1, 2.4, 2.0, 2.3, 2.0, 2.1, 2.3, 2.2],
    "nloc": [11, 13, 10, 12, 8, 9, 7, 10],
    "tokens": [82, 91, 76, 88, 55, 64, 58, 61],
    "parameters": [1, 1, 2, 1, 1, 0, 1, 1],
    "function_length": [13, 14, 11, 12, 9, 10, 8, 11],
    "ccn_density": [0.25, 0.22, 0.27, 0.24, 0.30, 0.31, 0.29, 0.28]
})

def cliffs_delta(x, y):
    n_greater = 0
    n_less = 0

    for xi in x:
        for yi in y:
            if xi > yi:
                n_greater += 1
            elif xi < yi:
                n_less += 1

    return (n_greater - n_less) / (len(x) * len(y))


metrics = [
    "ccn",
    "nloc",
    "tokens",
    "parameters",
    "function_length",
    "ccn_density"
]

results = []

for metric in metrics:
    ai = data[data["source"] == "AI"][metric]
    human = data[data["source"] == "Human"][metric]

    stat, p = mannwhitneyu(ai, human, alternative="two-sided")
    delta = cliffs_delta(ai, human)

    results.append({
        "Metric": metric,
        "AI Mean": ai.mean(),
        "Human Mean": human.mean(),
        "Mann-Whitney U": stat,
        "p-value": p,
        "Cliff's Delta": delta
    })

results_df = pd.DataFrame(results)
print(results_df)