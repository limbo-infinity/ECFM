import argparse
import csv
import glob
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TIME_FORMATS = ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
NYISO_ZONES = (
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
)


def parse_timestamp(value):
    for time_format in TIME_FORMATS:
        try:
            return datetime.strptime(value, time_format)
        except ValueError:
            pass

    raise ValueError(f"Could not parse timestamp: {value!r}")


def expand_paths(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]

    expanded = []
    for path in paths:
        matches = sorted(glob.glob(str(path)))
        expanded.extend(matches if matches else [str(path)])

    return expanded


def load_lbmp_csv(
    paths,
    price_column="LBMP ($/MWHr)",
    timestamp_column="Time Stamp",
    zone_column="Name",
    zones=None,
    include_trade_sides=False,
):
    """
    Parse NYISO LBMP CSV files into daily price vectors.

    Returns:
        prices: [num_days, K]
        metadata: dict with dates, zones, hours, and K
    """
    daily_prices = defaultdict(dict)
    seen_zones = set()
    seen_hours = set()

    for path in expand_paths(paths):
        with open(path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                timestamp = parse_timestamp(row[timestamp_column])
                date = timestamp.date()
                hour = timestamp.hour
                zone = row[zone_column].strip()
                price = float(row[price_column])

                daily_prices[date][(zone, hour)] = price
                seen_zones.add(zone)
                seen_hours.add(hour)

    if not daily_prices:
        raise ValueError("No price rows were loaded from the CSV file(s).")

    zones = list(zones) if zones is not None else sorted(seen_zones)
    hours = list(range(24))
    dates = sorted(daily_prices)

    price_rows = []
    for date in dates:
        values = []
        missing = []

        for zone in zones:
            for hour in hours:
                key = (zone, hour)
                if key not in daily_prices[date]:
                    missing.append(f"{zone} hour {hour}")
                    continue
                values.append(daily_prices[date][key])
        
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise ValueError(
                f"{date} is missing {len(missing)} zone-hour prices "
                f"({missing_preview})."
            )

        price_rows.append(values)

    prices = np.asarray(price_rows, dtype=np.float32)

    if include_trade_sides:
        prices = np.concatenate([prices, prices], axis=1)

    metadata = {
        "dates": dates,
        "zones": zones,
        "hours": hours,
        "num_days": len(dates),
        "num_zones": len(zones),
        "hours_per_day": len(hours),
        "include_trade_sides": include_trade_sides,
        "K": prices.shape[1],
    }
    return prices, metadata


class VirtualTradingDataset(Dataset):
    def __init__(self, da_prices, rt_prices, history_len=30, gap=2, target_indices=None):
        """
        da_prices: [num_days, K]
        rt_prices: [num_days, K]
        """
        self.da_prices = torch.as_tensor(da_prices, dtype=torch.float32)
        self.rt_prices = torch.as_tensor(rt_prices, dtype=torch.float32)
        self.history_len = history_len
        self.gap = gap

        if self.da_prices.shape != self.rt_prices.shape:
            raise ValueError("da_prices and rt_prices must have the same shape.")

        self.num_days, self.K = self.da_prices.shape
        first_valid_t = history_len + gap - 1

        if first_valid_t >= self.num_days:
            raise ValueError(
                "Not enough days to build a dataset. Need at least "
                f"{first_valid_t + 1} days for history_len={history_len} and gap={gap}."
            )

        if target_indices is None:
            self.target_indices = list(range(first_valid_t, self.num_days))
        else:
            self.target_indices = list(target_indices)

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        target_day = self.target_indices[idx]
        latest_input_day = target_day - self.gap
        
        start = latest_input_day - self.history_len + 1
        end = latest_input_day + 1
        hist_da = self.da_prices[start:end]
        hist_rt = self.rt_prices[start:end]
        
        xi = torch.cat([hist_da, hist_rt], dim=0).reshape(-1)
        da_true = self.da_prices[target_day].unsqueeze(0)
        rt_true = self.rt_prices[target_day].unsqueeze(0)

        return xi, da_true, rt_true


def make_train_val_loaders(
    da_prices,
    rt_prices,
    history_len=30,
    gap=2,
    batch_size=128,
    train_frac=0.8,
):
    num_days = da_prices.shape[0]
    first_valid_t = history_len + gap - 1
    all_target_days = np.arange(first_valid_t, num_days)
    split_day = int(train_frac * num_days)
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
        shuffle=False,
    )
    return train_loader, val_loader


def load_train_val_from_csv(
    da_paths,
    rt_paths,
    history_len=30,
    gap=2,
    batch_size=128,
    train_frac=0.8,
    include_trade_sides=False,
    zones=None,
):
    da_prices, da_metadata = load_lbmp_csv(
        da_paths,
        zones=zones,
        include_trade_sides=include_trade_sides,
    )
    rt_prices, rt_metadata = load_lbmp_csv(
        rt_paths,
        zones=da_metadata["zones"],
        include_trade_sides=include_trade_sides,
    )

    if da_metadata["dates"] != rt_metadata["dates"]:
        raise ValueError("Day-ahead and real-time CSV files must cover the same dates.")

    train_loader, val_loader = make_train_val_loaders(
        da_prices=da_prices,
        rt_prices=rt_prices,
        history_len=history_len,
        gap=gap,
        batch_size=batch_size,
        train_frac=train_frac,
    )
    return train_loader, val_loader, da_metadata


def main():
    parser = argparse.ArgumentParser(description="Parse NYISO LBMP CSV files.")
    parser.add_argument("da_csv", help="Day-ahead LBMP CSV path or glob.")
    parser.add_argument("--rt-csv", help="Real-time LBMP CSV path or glob.")
    parser.add_argument("--include-trade-sides", action="store_true")
    parser.add_argument(
        "--nyiso-zones-only",
        action="store_true",
        help="Use the 11 internal NYISO load zones and exclude external proxy zones.",
    )
    args = parser.parse_args()
    zones = NYISO_ZONES if args.nyiso_zones_only else None

    da_prices, metadata = load_lbmp_csv(
        args.da_csv,
        zones=zones,
        include_trade_sides=args.include_trade_sides,
    )
    print("Loaded day-ahead prices")
    print(f"days: {metadata['num_days']}")
    print(f"zones: {metadata['num_zones']}")
    print(f"hours/day: {metadata['hours_per_day']}")
    print(f"K: {metadata['K']}")
    print(f"shape: {da_prices.shape}")

    if args.rt_csv:
        rt_prices, _ = load_lbmp_csv(
            args.rt_csv,
            zones=metadata["zones"],
            include_trade_sides=args.include_trade_sides,
        )
        print("Loaded real-time prices")
        print(f"shape: {rt_prices.shape}")


if __name__ == "__main__":
    main()
