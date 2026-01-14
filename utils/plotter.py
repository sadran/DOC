import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, save_plots: bool = False, save_dir: str = None):
        self.save_plots = save_plots
        self.save_dir = save_dir
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
            ax.scatter([n] * len(errors), errors, color='red', alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.figures.append((figure, ax))
        return figure, ax

    def plot_doc_vs_erm(self, n_values, erm_means, doc_means, title="Mean True Error: ERM vs DOC"):
        fig, ax = plt.subplots(figsize=(6, 4))

        # red x: empirical mean test error of ERM solutions
        ax.plot(n_values, erm_means, "x", c="blue", label="Empirical mean (ERM solutions)")

        # blue +: DOC-based predicted mean
        ax.plot(n_values, doc_means, "+", c="red", label="DOC-based bound/prediction")

        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("En")
        ax.legend()
        return fig, ax

    def show_plots(self):
        for figure, _ in self.figures:
            plt.figure(figure.number)   # activate the figure
            plt.show()