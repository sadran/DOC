import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, save_plots: bool = False, save_dir: str = None):
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.figures = []
        
    def plot_histogram(self, 
                       data, bins=100, 
                       density=True, 
                       title="Histogram",
                       xlabel="Value", 
                       ylabel="Density"):
        figure, ax = plt.subplots()
        ax.hist(data, bins=bins, density=density)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.figures.append((figure, ax))

    def show_plots(self):
        for figure, _ in self.figures:
            plt.figure(figure.number)   # activate the figure
            plt.show()