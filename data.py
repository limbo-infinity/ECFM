import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Fake dataset to test on

num_days = 2000          # fake number of historical days
N = 11                   # NYISO has 11 zones
hours_per_day = 24

# If using both virtual demand and virtual supply:
K = 2 * N * hours_per_day   # 528 options

# Base prices
rng = np.random.default_rng(seed=42)

# Shape: [num_days, K]
da_prices = 30 + 10 * rng.standard_normal(size=(num_days, K))
rt_prices = da_prices + 5 * rng.standard_normal(size=(num_days, K))

# Add occasional price spikes
spike_mask = rng.random(size=(num_days, K)) < 0.01
rt_prices[spike_mask] += rng.normal(loc=100, scale=30, size=spike_mask.sum())

# Clip to NYISO-style bounds
da_prices = np.clip(da_prices, 0, 1000)
rt_prices = np.clip(rt_prices, 0, 1000)

print("da_prices shape:", da_prices.shape)
print("rt_prices shape:", rt_prices.shape)

class VirtualTradingDataset(Dataset):
    def __init__(self, da_prices, rt_prices, history_len=30, gap=2, target_indices=None):
        """
        da_prices: [num_days, K] or [num_days, N, 24]
        rt_prices: [num_days, K] or [num_days, N, 24]

        history_len:
            number of past days used as input

        gap:
            gap=2 means to predict/trade day t,
            the most recent usable observation is day t-2

        target_indices:
            list/array of target days t
        """

        da_prices = torch.tensor(da_prices, dtype=torch.float32)
        rt_prices = torch.tensor(rt_prices, dtype=torch.float32)

        # If data is [days, zones, hours], flatten zones and hours into K
        if da_prices.ndim == 3:
            D, N, H = da_prices.shape
            da_prices = da_prices.reshape(D, N * H)
            rt_prices = rt_prices.reshape(D, N * H)

        self.da_prices = da_prices
        self.rt_prices = rt_prices
        self.history_len = history_len
        self.gap = gap

        self.num_days, self.K = da_prices.shape

        if target_indices is None:
            first_valid_t = history_len + gap - 1
            self.target_indices = list(range(first_valid_t, self.num_days))
        else:
            self.target_indices = list(target_indices)

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        t = self.target_indices[idx]

        # Most recent usable day is t - gap
        # With gap=2, newest input is day t-2
        latest_input_day = t - self.gap

        start = latest_input_day - self.history_len + 1
        end = latest_input_day + 1

        hist_da = self.da_prices[start:end]  # [history_len, K]
        hist_rt = self.rt_prices[start:end]  # [history_len, K]

        # Flatten history into one feature vector
        xi = torch.cat([hist_da, hist_rt], dim=0).reshape(-1) 
        #Understand how each method flattens it 

        # Target prices for day t
        da_true = self.da_prices[t].unsqueeze(0)  # [1, K]
        rt_true = self.rt_prices[t].unsqueeze(0)  # [1, K]

        return xi, da_true, rt_true

# Example:
# da_prices.shape = [num_days, K]
# rt_prices.shape = [num_days, K]

history_len = 30
gap = 2
batch_size = 128



num_days = da_prices.shape[0]

first_valid_t = history_len + gap - 1
all_target_days = np.arange(first_valid_t, num_days)

# Chronological split to avoid future leakage
split_day = int(0.8 * num_days)

train_target_days = all_target_days[all_target_days < split_day]
val_target_days = all_target_days[all_target_days >= split_day]

train_dataset = VirtualTradingDataset(
    da_prices=da_prices,
    rt_prices=rt_prices,
    history_len=history_len,
    gap=gap,
    target_indices=train_target_days,
)

val_dataset = VirtualTradingDataset(
    da_prices=da_prices,
    rt_prices=rt_prices,
    history_len=history_len,
    gap=gap,
    target_indices=val_target_days,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,)