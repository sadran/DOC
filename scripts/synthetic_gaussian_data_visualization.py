from matplotlib import pyplot as plt
from core.dataset import Gaussian

def visualize(data, label):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap='coolwarm', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title('2D Data Visualization')
    plt.savefig('scripts/synthetic_gaussian_data_visualization.png')

def main():
    dataset = Gaussian(feature_dim=2, n_samples_per_class=5000, mean_distance=2.0, sigma=0.5, seed=42)
    data = dataset.x
    label = dataset.y
    visualize(data, label)

if __name__ == "__main__":
    main()