import torch
import torch.nn as nn

# We input z ~ N(0, I) conditioned on the DA and RT prices
class ConditionalFlow(nn.Module):
    def __init__(self, T, K, hidden_dim):
        super().__init__()
        self.T = T
        self.K = K

        x_dim = T * K
        cond_dim = 2 * T * K

        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, z, t, da_cond, rt_cond):
        """
        z:       [batch, T, K]
        t:       [batch, 1]
        da_cond: [batch, T, K]
        rt_cond: [batch, T, K]
        """

        batch_size = z.shape[0]

        z_flat = z.reshape(batch_size, -1)
        cond_flat = torch.cat(
            [da_cond.reshape(batch_size, -1),
             rt_cond.reshape(batch_size, -1)],
            dim=-1,
        )

        inp = torch.cat([z_flat, cond_flat, t], dim=-1)

        velocity = self.net(inp)
        velocity = velocity.reshape(batch_size, self.T, self.K)

        return velocity
