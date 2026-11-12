import torch
class Metric(torch.nn.Module):
    def reset(self):
        self.conf_matrix.fill(0)
        self.conf_matrix_change.fill(0)
        self.conf_matrix_sc.fill(0)
