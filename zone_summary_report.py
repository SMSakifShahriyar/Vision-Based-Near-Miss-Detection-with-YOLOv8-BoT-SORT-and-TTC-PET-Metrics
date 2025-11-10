import pandas as pd

df = pd.read_csv(r"F:\MY PROJECTS\VIDEO\conflicts_with_zones.csv")

# counts by zone and type
summary = df.pivot_table(index="zone", columns="type", values="t", aggfunc="count", fill_value=0)
summary["total"] = summary.sum(axis=1)
summary.loc["total"] = summary.sum()

print(summary)
