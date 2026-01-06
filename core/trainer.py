class Trainer:
    def __init__(self):
        ...
    def train(self):  # Empirical risk minimization loop
        ...
    def train_until_zero_error(self):
        ...
    def sample_unit_sphere_weights_until_zero_error(self, model, train_loader, evaluator):  # For random sampling experiments
        device = evaluator.device
        while True:
            flat_weights = model.sample_unit_sphere_weights(device=device)
            model.set_flatten_weights(flat_weights)
            train_error = evaluator.compute_error(model, train_loader)
            if train_error == 0.0:
                break  