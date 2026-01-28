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
            ax.scatter([n] * len(errors), errors, color='red', alpha=0.5)

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
        ax.plot(n_values, erm_means, "x", c="blue", label="Empirical mean (ERM solutions)")

        # blue +: DOC-based predicted mean
        ax.plot(xticks, doc_means, "+", c="red", label="DOC-based bound/prediction")

        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("En")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xticks(xticks)
        ax.legend()
        return fig, ax

    def show_plots(self):
        for figure, _ in self.figures:
            plt.figure(figure.number)   # activate the figure
            plt.show()