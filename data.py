import csv
import glob
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TIME_FORMATS = (
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)
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
ZONE_DATA_DIR = Path("zone_data")
TIMESTAMP_COLUMNS = ("Time Stamp", "Eastern Date Hour", "RTD End Time Stamp")
ZONE_COLUMNS = ("Name", "Zone Name")
PRICE_COLUMNS = ("LBMP ($/MWHr)", "DAM Zonal LBMP", "RTD Zonal LBMP")


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


def resolve_column(fieldnames, requested, options):
    if requested in fieldnames:
        return requested

    for option in options:
        if option in fieldnames:
            return option

    raise ValueError(
        f"Could not find any of these columns: {', '.join(options)}. "
        f"Found: {', '.join(fieldnames)}"
    )


def load_lbmp_csv(
    paths,
    price_column=None,
    timestamp_column=None,
    zone_column=None,
    zones=None,
    include_trade_sides=False,
    dates=None,
    drop_incomplete_days=False,
):
    """
    Parse NYISO LBMP CSV files into daily price vectors.

    Returns:
        prices: [num_days, K]
        metadata: dict with dates, zones, hours, and K
    """
    daily_price_totals = defaultdict(lambda: defaultdict(float))
    daily_price_counts = defaultdict(lambda: defaultdict(int))
    seen_zones = set()
    seen_hours = set()
    dates = set(dates) if dates is not None else None

    for path in expand_paths(paths):
        with open(path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            resolved_timestamp_column = resolve_column(
                fieldnames,
                timestamp_column,
                TIMESTAMP_COLUMNS,
            )
            resolved_zone_column = resolve_column(fieldnames, zone_column, ZONE_COLUMNS)
            resolved_price_column = resolve_column(fieldnames, price_column, PRICE_COLUMNS)

            for row in reader:
                timestamp = parse_timestamp(row[resolved_timestamp_column])
                date = timestamp.date()
                if dates is not None and date not in dates:
                    continue

                hour = timestamp.hour
                zone = row[resolved_zone_column].strip()
                price = float(row[resolved_price_column])

                key = (zone, hour)
                daily_price_totals[date][key] += price
                daily_price_counts[date][key] += 1
                seen_zones.add(zone)
                seen_hours.add(hour)

    if not daily_price_totals:
        raise ValueError("No price rows were loaded from the CSV file(s).")

    daily_prices = defaultdict(dict)
    for date, totals in daily_price_totals.items():
        for key, total in totals.items():
            daily_prices[date][key] = total / daily_price_counts[date][key]

    zones = list(zones) if zones is not None else sorted(seen_zones)
    hours = list(range(24))
    dates = sorted(daily_prices)
    skipped_dates = []
    loaded_dates = []

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
            if drop_incomplete_days:
                skipped_dates.append(date)
                continue

            missing_preview = ", ".join(missing[:5])
            raise ValueError(
                f"{date} is missing {len(missing)} zone-hour prices "
                f"({missing_preview})."
            )

        price_rows.append(values)
        loaded_dates.append(date)

    prices = np.asarray(price_rows, dtype=np.float32)
    if prices.size == 0:
        raise ValueError("No complete daily price rows were loaded from the CSV file(s).")

    if include_trade_sides:
        prices = np.concatenate([prices, prices], axis=1)

    metadata = {
        "dates": loaded_dates,
        "zones": zones,
        "hours": hours,
        "num_days": len(loaded_dates),
        "num_zones": len(zones),
        "hours_per_day": len(hours),
        "include_trade_sides": include_trade_sides,
        "skipped_dates": skipped_dates,
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
        first_valid_t = history_len + gap - 1  ##Not sure if this is right
        

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
        
        ## don't assume that we start from day 1 but rather a slice
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
    train_drop_last=False,
):
    num_days = da_prices.shape[0]
    first_valid_t = history_len + gap - 1 ## why are we starting from hist len + gap
    all_target_days = np.arange(first_valid_t, num_days)
    train_days = int(train_frac * num_days)
    train_target_days = all_target_days[all_target_days < train_days]
    val_target_days = all_target_days[all_target_days >= train_days]
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
        drop_last=train_drop_last,
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
    train_drop_last=False,
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
        dates=da_metadata["dates"],
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
        train_drop_last=train_drop_last,
    )
    return train_loader, val_loader, da_metadata


def load_prices_from_zone_data(
    zone_data_dir=ZONE_DATA_DIR,
    include_trade_sides=False,
    zones=NYISO_ZONES,
    drop_incomplete_days=True,
):
    zone_data_dir = Path(zone_data_dir)
    da_paths = str(zone_data_dir / "DA" / "**" / "*.csv")
    rt_paths = str(zone_data_dir / "RT" / "**" / "*.csv")

    da_prices, da_metadata = load_lbmp_csv(
        da_paths,
        include_trade_sides=include_trade_sides,
        zones=zones,
        drop_incomplete_days=drop_incomplete_days,
    )
    rt_prices, rt_metadata = load_lbmp_csv(
        rt_paths,
        include_trade_sides=include_trade_sides,
        zones=da_metadata["zones"],
        dates=da_metadata["dates"],
        drop_incomplete_days=drop_incomplete_days,
    )

    if da_metadata["dates"] != rt_metadata["dates"]:
        raise ValueError("Day-ahead and real-time CSV files must cover the same dates.")

    return da_prices, rt_prices, da_metadata


def load_train_val_from_zone_data(
    zone_data_dir=ZONE_DATA_DIR,
    history_len=30,
    gap=2,
    batch_size=128,
    train_frac=0.8,
    train_drop_last=False,
    include_trade_sides=False,
    zones=NYISO_ZONES,
):
    da_prices, rt_prices, metadata = load_prices_from_zone_data(
        zone_data_dir=zone_data_dir,
        include_trade_sides=include_trade_sides,
        zones=zones,
    )
    train_loader, val_loader = make_train_val_loaders(
        da_prices=da_prices,
        rt_prices=rt_prices,
        history_len=history_len,
        gap=gap,
        batch_size=batch_size,
        train_frac=train_frac,
        train_drop_last=train_drop_last,
    )
    return train_loader, val_loader, metadata


def main():
    history_len = 30
    gap = 2
    batch_size = 128
    train_frac = 0.8
    include_trade_sides = False

    da_prices, rt_prices, metadata = load_prices_from_zone_data(
        include_trade_sides=include_trade_sides,
    )
    print("Loaded prices from zone_data")
    print(f"days: {metadata['num_days']}")
    print(f"zones: {metadata['num_zones']}")
    print(f"hours/day: {metadata['hours_per_day']}")
    print(f"K: {metadata['K']}")
    print(f"day-ahead shape: {da_prices.shape}")
    print(f"real-time shape: {rt_prices.shape}")

    try:
        train_loader, val_loader = make_train_val_loaders(
            da_prices=da_prices,
            rt_prices=rt_prices,
            history_len=history_len,
            gap=gap,
            batch_size=batch_size,
            train_frac=train_frac,
        )
    except ValueError as error:
        print(f"Could not create training and validation sets: {error}")
        return

    print("Created training and validation sets")
    print(f"train batches: {len(train_loader)}")
    print(f"validation batches: {len(val_loader)}")


if __name__ == "__main__":
    main()
