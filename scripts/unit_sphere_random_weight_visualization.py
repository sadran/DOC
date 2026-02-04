import torch 
from models.mlp import MLP
from matplotlib import pyplot as plt

def visualize_weights(x_weights: torch.Tensor, y_weights: torch.Tensor, axis_lim: tuple, title: str, save_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(x_weights, y_weights, alpha=0.7)
    plt.xlabel('Weight Dimension 1')
    plt.ylabel('Weight Dimension 2')
    plt.xlim(axis_lim[0], axis_lim[1])
    plt.ylim(axis_lim[0], axis_lim[1])
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)

def main():
    model1 = MLP(
        input_dim=1,
        hidden_layers=[],
        output_dim=2,
        activation="leaky_relu",
        bias=False,
    )
    weights = [model1.sample_unit_sphere_weights(device='cpu') for _ in range(1000)]
    weights = torch.stack(weights)  
    visualize_weights(weights[:, 0], 
                      weights[:, 1], 
                      axis_lim=(-1.5, 1.5),
                      title='1000 weight vectors with 2 parameters sampled from unit sphere', 
                      save_path='scripts/random_weights_1000_vector_2_param.png')


    model2 = MLP(
        input_dim=10,
        hidden_layers=[100],
        output_dim=10,
        activation="leaky_relu",
        bias=False,
    )
    weights = model2.sample_unit_sphere_weights(device='cpu')
    weights = weights.view(-1, 2)
    visualize_weights(weights[:, 0], 
                      weights[:, 1], 
                      axis_lim=(-0.2, 0.2),
                      title=f'1 weith vector with {model2.num_parameters()} parameters\n sampled from unit sphere\n visualized in 2D', 
                      save_path=f'scripts/random_weights_1_vector_{model2.num_parameters()}_param.png')

if __name__ == "__main__":
    main()