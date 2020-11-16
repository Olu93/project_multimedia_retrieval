from helper.config import STAT_PATH
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv(STAT_PATH + "\\orig_stats.csv")
    co = Counter(df["label"])
    plt.figure(figsize=(25, 13))
    plt.xlabel("Classes")
    plt.ylabel("Number of instances")
    plt.xticks(rotation=90)
    plt.yticks(range(0, len(co.keys()), 5))
    plt.title("Number of instance per class.")
    plt.bar([key.replace("_", " ").title() for key in list(co.keys())], co.values(), align='edge')
    plt.savefig("figs\\instance_per_class.png")
    plt.show()
