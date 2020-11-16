# %%

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

# def get_top_weights(df, class_label):
#     top_ant_val = df[df.class_label == class_label].val.max()
#     top_features = df[df.val > .90 * top_ant_val]
#     top_features_counts = top_features.sum()
#     top_weights = top_features_counts.iloc[:-2].values
#     weights_w_strong = ([2.9]) + ([134] * count_hists) + ([1.58] * count_skeletons)
#     return tuple(top_weights * weights_w_strong)

if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    data = pd.read_csv('stats/hyper_params_copy.csv', index_col=False)
    
    
    # best_weight_combos_data = pd.read_csv('feature_combinations.csv')



    # top_ant_weights = get_top_weights(best_weight_combos_data, "ant")
    # top_helmet_weights = get_top_weights(best_weight_combos_data, "helmet")
    
    
    
    grouped = data.groupby("sr hr skr".split()).mean()
    points = grouped.reset_index().values
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
    cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(points[:, 3].min(), points[:, 3].max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fs = 9
    cbar = fig.colorbar(sm, shrink=0.8, pad=0.1)
    cbar.ax.set_title("Adjusted F1-Score", fontsize=fs)
    ax.set_xlabel("Scalar Vector Weights", fontsize=fs)
    ax.set_ylabel("Histogram Feature Weights", fontsize=fs)
    ax.set_zlabel("2D Descriptor Feature weights", fontsize=fs)
    # ax.xaxis._axinfo['label']['space_factor'] = 2.8
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0., right=0.75, top=1., wspace=1)
    plt.savefig("figs/fig_hyperparam_opt.png")
    plt.show()
