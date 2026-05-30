import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ExperimentLogger:
    def __init__(self, root_dir="experiments", run_name="run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in run_name.strip()
        )
        self.run_dir = Path(root_dir) / f"{timestamp}_{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def path(self, filename):
        return self.run_dir / filename

    def save_json(self, filename, payload):
        output_path = self.path(filename)
        with open(output_path, "w") as output_file:
            json.dump(payload, output_file, indent=2, default=_json_default)
        return output_path

    def save_config(self, config):
        return self.save_json("config.json", config)

    def save_summary(self, summary):
        return self.save_json("summary.json", summary)

    def save_metrics_csv(self, filename, metrics):
        output_path = self.path(filename)
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["metric", "value"])
            for key in sorted(metrics):
                writer.writerow([key, metrics[key]])
        return output_path

    def save_npz(self, filename, **arrays):
        output_path = self.path(filename)
        np.savez_compressed(output_path, **arrays)
        return output_path

    def save_model(self, filename, model, extra=None):
        output_path = self.path(filename)
        payload = {"state_dict": model.state_dict()}
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, output_path)
        return output_path
