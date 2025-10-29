import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

true_data = np.load(
    '/home/haitong/PycharmProjects/diffusion-model-hallucination/gaussian_experiments/chkpts/UnbalancedGaussian2D_200000_g_1_e_300_t1000_m128_nl3_blinear_seed1234_0/real_dataset.npy')


def plot_hist(gen_data, title, fig_name, original_data = False):
    np.random.shuffle(gen_data)
    gen_data = gen_data[:2000]

    if not original_data:
        # Estimate distributions using histograms
        bins = 50
        hist1, bin_edges = np.histogram(gen_data, bins=bins, density=True)
        hist2, _ = np.histogram(true_data, bins=bin_edges, density=True)

        # Avoid division by zero
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize histograms to get probabilities
        prob1 = hist1 / np.sum(hist1)
        prob2 = hist2 / np.sum(hist2)

        # Compute KL divergence
        kl_divergence = entropy(prob1, prob2)
        print("KL Divergence:", kl_divergence)

    plt.figure(figsize=(3, 3))
    # Generate example 2D data
    x = gen_data[:, 0]
    y = gen_data[:, 1]

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(3, 3))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    # Scatter plot
    scatter_ax = fig.add_subplot(grid[1:, :-1])
    scatter_ax.scatter(x, y, alpha=0.5, s=0.5)
    scatter_ax.set_xlim([-6, 6])
    scatter_ax.set_ylim([-6, 6])
    scatter_ax.set_xticks(np.array([-6, -3, 0, 3, 6]))
    scatter_ax.set_yticks(np.array([-6, -3, 0, 3, 6]))
    scatter_ax.grid(True)

    # Histogram for X-axis
    x_hist_ax = fig.add_subplot(grid[0, :-1], sharex=scatter_ax)
    counts, bins, patches = x_hist_ax.hist(x, bins=[-6, 0, 6], color='blue', alpha=0.7,
                                           weights=np.ones_like(x) / len(x))
    # x_hist_ax.axis('off')  # Hide x-ticks and labels
    x_hist_ax.grid(True)
    x_hist_ax.set_yticks([0.2, 0.8, ])
    x_hist_ax.set_yticklabels(['20%', '80%'])
    x_hist_ax.tick_params(axis='x', which='both', labelbottom=False)

    # Add numbers to the bins
    for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last edge since bins have one extra
        bin_center = bin_edge + (bins[1] - bins[0]) / 2  # Calculate the center of the bin
        plt.text(bin_center, count - 0.2, f'{count:.0%}', ha='center', va='bottom', color='white', fontsize=8)

    # Histogram for Y-axis
    y_hist_ax = fig.add_subplot(grid[1:, -1], sharey=scatter_ax)
    counts, bins, patches = y_hist_ax.hist(y, bins=[-6, 0, 6], orientation='horizontal', color='green', alpha=0.7,
                                           weights=np.ones_like(y) / len(y))
    y_hist_ax.grid(True)
    y_hist_ax.tick_params(axis='y', which='both', labelleft=False)
    y_hist_ax.set_xticks([0.2, 0.8])
    y_hist_ax.set_xticklabels(['20%', '80%'])

    # Add numbers to the bins
    for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last edge since bins have one extra
        bin_center = bin_edge + (bins[1] - bins[0]) / 2  # Calculate the center of the bin
        plt.text(count - 0.1, bin_center, f'{count:.0%}', ha='center', va='bottom', color='black', fontsize=8)

    if not original_data:
        plt.suptitle(f"{title}\n KL Div to true data: {kl_divergence:.3f}", fontsize=10)
    else:
        plt.suptitle(title, fontsize=14)

    # Adjust the layout to avoid overlaps
    plt.tight_layout()

    # Show the plot
    plt.savefig(fig_name, dpi=300)


def plot_scatter_with_custom_true_data(gen_data, title, fig_name, true_data=None, original_data = False):
    np.random.shuffle(gen_data)
    gen_data = gen_data[:2000]

    if true_data is not None:
        # Estimate distributions using histograms
        bins = 100
        hist1, bin_edges = np.histogram(gen_data, bins=bins, density=True)
        hist2, _ = np.histogram(true_data, bins=bin_edges, density=True)

        # Avoid division by zero
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize histograms to get probabilities
        prob1 = hist1 / np.sum(hist1)
        prob2 = hist2 / np.sum(hist2)

        # Compute KL divergence
        kl_divergence = entropy(prob1, prob2)
        print("KL Divergence:", kl_divergence)

    plt.figure(figsize=(3, 3))
    # Generate example 2D data
    x = gen_data[:, 0]
    y = gen_data[:, 1]

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(3, 3))
    # grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    # Scatter plot
    scatter_ax = fig.add_subplot()
    scatter_ax.scatter(x, y, alpha=0.5, s=0.5)
    scatter_ax.set_xlim([-3, 3])
    scatter_ax.set_ylim([-3, 3])
    # scatter_ax.set_xticks(np.array([-6, -3, 0, 3, 6]))
    # scatter_ax.set_yticks(np.array([-6, -3, 0, 3, 6]))
    scatter_ax.grid(True)

    # # Histogram for X-axis
    # x_hist_ax = fig.add_subplot(grid[0, :-1], sharex=scatter_ax)
    # counts, bins, patches = x_hist_ax.hist(x, bins=[-6, 0, 6], color='blue', alpha=0.7,
    #                                        weights=np.ones_like(x) / len(x))
    # # x_hist_ax.axis('off')  # Hide x-ticks and labels
    # x_hist_ax.grid(True)
    # x_hist_ax.set_yticks([0.2, 0.8, ])
    # x_hist_ax.set_yticklabels(['20%', '80%'])
    # x_hist_ax.tick_params(axis='x', which='both', labelbottom=False)

    # # Add numbers to the bins
    # for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last edge since bins have one extra
    #     bin_center = bin_edge + (bins[1] - bins[0]) / 2  # Calculate the center of the bin
    #     plt.text(bin_center, count - 0.2, f'{count:.0%}', ha='center', va='bottom', color='white', fontsize=8)
    #
    # # Histogram for Y-axis
    # y_hist_ax = fig.add_subplot(grid[1:, -1], sharey=scatter_ax)
    # counts, bins, patches = y_hist_ax.hist(y, bins=[-6, 0, 6], orientation='horizontal', color='green', alpha=0.7,
    #                                        weights=np.ones_like(y) / len(y))
    # y_hist_ax.grid(True)
    # y_hist_ax.tick_params(axis='y', which='both', labelleft=False)
    # y_hist_ax.set_xticks([0.2, 0.8])
    # y_hist_ax.set_xticklabels(['20%', '80%'])
    #
    # # Add numbers to the bins
    # for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last edge since bins have one extra
    #     bin_center = bin_edge + (bins[1] - bins[0]) / 2  # Calculate the center of the bin
    #     plt.text(count - 0.1, bin_center, f'{count:.0%}', ha='center', va='bottom', color='black', fontsize=8)

    if true_data is not None:
        plt.title(f"{title}\n KL Div to true data: {kl_divergence:.3e}", fontsize=10)
    else:
        plt.title(title, fontsize=14)

    # Adjust the layout to avoid overlaps
    plt.tight_layout()

    # Show the plot
    plt.savefig(fig_name, dpi=300)