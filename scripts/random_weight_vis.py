import torch
from matplotlib import pyplot as plt

def visualize(data, label):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap='coolwarm', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title('2D Data Visualization')
    plt.savefig('scripts/random_weight_visualization.png')

def main():
    # data parameters
    n_samples_per_class = 1000
    mean_distance = 2
    sigma = 0.5

    # two dimentional data
    mu0 = torch.zeros(2)
    mu1 = torch.zeros(2)
    # one class centered at [1, 0] and the other at [-1, 0]
    mu0[0] = mean_distance / 2
    mu1[0] = -mean_distance / 2

    x0 = torch.normal(mu0.repeat([n_samples_per_class, 1]), sigma)
    x1 = torch.normal(mu1.repeat([n_samples_per_class, 1]), sigma)

    y0 = torch.zeros(n_samples_per_class, dtype=torch.long)
    y1 = torch.ones(n_samples_per_class, dtype=torch.long)

    x = torch.vstack([x0, x1])
    y = torch.hstack([y0, y1])
    visualize(x, y)

if __name__ == "__main__":
    main()