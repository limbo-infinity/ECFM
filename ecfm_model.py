import torch
import torch.nn as nn
from helper import transform_x
import torch.nn.functional as F

# v(x_t, t, c)
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



    def rectified_flow_loss(self, z_1, da_cond, rt_cond):
        """
        Rectified-flow objective from noise z_0 to target sample z_1.

        z_1:     [batch, T, K]
        da_cond: [batch, T, K]
        rt_cond: [batch, T, K]
        """
        batch_size = z_1.shape[0]

        z_0 = torch.randn_like(z_1)
        t = z_1.new_empty(batch_size, 1).uniform_(0.0, 1.0)
        t_view = t.reshape(batch_size, 1, 1)

        z_t = (1.0 - t_view) * z_0 + t_view * z_1
        target_velocity = z_1 - z_0
        pred_velocity = self(z_t, t, da_cond, rt_cond)

        return ((pred_velocity - target_velocity) ** 2).mean()

## Sampler for the CFM
class ECFMModel(nn.Module):
    def __init__(self, T, K, hidden_dim):
        super().__init__()

        self.T = T
        self.K = K
        self.v_theta = ConditionalFlow(T=T, K=K, hidden_dim=hidden_dim)

    @torch.no_grad()
    def sample(self, da_cond, rt_cond, num_samples=1, num_steps=50):
        """
        da_cond: [batch, T, K]
        rt_cond: [batch, T, K]

        returns:
            x_candidates: [batch, num_samples, T, K]
        """

        batch_size, T, K = da_cond.shape
        device = da_cond.device

        # Repeat conditioning variables for multiple samples per input
        da_rep = da_cond.unsqueeze(1).repeat(1, num_samples, 1, 1)
        rt_rep = rt_cond.unsqueeze(1).repeat(1, num_samples, 1, 1)

        da_rep = da_rep.reshape(batch_size * num_samples, T, K)
        rt_rep = rt_rep.reshape(batch_size * num_samples, T, K)

        # Initial noise x_0 ~ N(0, I)
        x = torch.randn(batch_size * num_samples, T, K, device=device)

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_value = step / num_steps
            t = torch.full(
                (batch_size * num_samples, 1),
                t_value,
                device=device,
            )

            velocity = self.v_theta(x, t, da_rep, rt_rep)

            # Euler ODE step
            x = x + dt * velocity

        x = x.reshape(batch_size, num_samples, T, K)

        return x

    
    
## Energy fn is just the trading energy from (6)
def langevin_refine_trading(
    x,
    da_price,
    rt_price,
    budget,
    energy_fn,
    beta=1.0,
    step_size=1e-3,
    n_steps=20,
):
    x = x.clone().detach()

    for _ in range(n_steps):
        x.requires_grad_(True)

        energy, logs = energy_fn(x=x, da_price=da_price, rt_price=rt_price,
                                budget=budget,)

        grad_x = torch.autograd.grad(energy, x, create_graph=False)[0]

        noise = torch.randn_like(x)
        x_new = x - step_size * beta * grad_x + (2 * step_size) ** 0.5 * noise
        x = x_new.detach()

    return x