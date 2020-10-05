# %%
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/psb/statistics.csv")
data.head()
# %%
x = data["bound_xmax"] - data["bound_xmin"]
y = data["bound_ymax"] - data["bound_ymin"]
z = data["bound_zmax"] - data["bound_zmin"]
data["bound_volume"] = x * y * z

data.head()
# %%
upper_bound = data["bound_volume"].quantile(.99)
lower_bound = data["bound_volume"].quantile(.0)
no_outlier_data = data.loc[((data.bound_volume > lower_bound) & (data.bound_volume < upper_bound))]
no_outlier_data.bound_volume.hist(bins=100)
# %%
f"Mean is {data.bound_volume.mean()} and std {data.bound_volume.std()}"
# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].hist(data.bound_volume, bins=100)
axes[0].set_title("Volume distribution with outliers")
axes[1].hist(no_outlier_data.bound_volume, bins=100)
axes[1].set_title("Volume distribution without outliers")
plt.tight_layout()
plt.show()
# %%
