import torch

class ValueNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon  # evita divisioni per zero

    def update(self, values: torch.Tensor):
        # Welford's online update
        batch_mean = values.mean().item()
        batch_var = values.var(unbiased=False).item()
        batch_count = values.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, values: torch.Tensor):
        return (values - self.mean) / (self.var ** 0.5 + 1e-8)

    def denormalize(self, values: torch.Tensor):
        return values * (self.var ** 0.5 + 1e-8) + self.mean
