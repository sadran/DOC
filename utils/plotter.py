import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

class Plotter:
    def __init__(self):
        self.figures = []
        
    def plot_histogram(self, 
                       data, bins=100, 
                       title="Histogram",
                       xlabel="Value", 
                       ylabel="Density"):
        figure, ax = plt.subplots()
        ax.hist(data, bins=bins)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.figures.append((figure, ax))
        return figure, ax

    def plot_boxplot(self, true_errors, n_values, title, xlabel, ylabel):
        figure, ax = plt.subplots()
        
        ax.boxplot(true_errors, labels=n_values, positions=n_values)
        for n, errors in zip(n_values, true_errors):
            ax.scatter([n] * len(errors), errors, marker='x', c='red', s=10, alpha=0.5)

        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.figures.append((figure, ax))
        return figure, ax

    def plot_doc_vs_erm(self, n_values, erm_means, doc_means, title="Mean True Error: ERM vs DOC"):
        xticks = [n for n in range(0, 31, 2)]

        fig, ax = plt.subplots(figsize=(6, 4))
        # red x: empirical mean test error of ERM solutions
        ax.plot(n_values, erm_means, "x", c="red", label="Empirical mean (ERM solutions)")

        # blue +: DOC-based predicted mean
        ax.plot(xticks, doc_means, "+", c="blue", label="DOC-based bound/prediction")

        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("En")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xticks(xticks)
        ax.legend()
        return fig, ax

    def plot_accuracy_curve(n_values, true_errors, use_std, label1="Model 1"):
        def compute_stats(true_errors):
            means = np.array([np.mean(r) for r in true_errors])

            if use_std:
                stds = np.array([np.std(r) for r in true_errors])
                err_low = stds
                err_high = stds
            else:
                mins = np.array([np.min(r) for r in true_errors])
                maxs = np.array([np.max(r) for r in true_errors])
                err_low = means - mins
                err_high = maxs - means
            return means, err_low, err_high

        x = np.array(n_values)

        fig, ax = plt.subplots(figsize=(8, 6))

        # ---- First model ----
        m1, l1, h1 = compute_stats(true_errors)
        ax.errorbar(x, m1, yerr=[l1, h1], fmt="--", linewidth=2, capsize=5, label=label1)

        ax.set_xlabel("n")
        ax.set_ylabel("Test accuracy")
        ax.grid(True, alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.show()
        return fig, ax

    def show_plots(self):
        for figure, _ in self.figures:
            plt.figure(figure.number)   # activate the figure
            plt.show()