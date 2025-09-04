# metrics.py
# Utility for logging run-time and usage metrics for Flavor Builder

import csv, time, os
from pathlib import Path
from typing import Dict, Any

LOG_FILE = Path(__file__).parent / "metrics_log.csv"

def log_metrics(
    query: str,
    runtime: float,
    shortlist_size: int,
    blend_size: int,
    metrics: Dict[str, Any]
):
    """
    Append a row of metrics to metrics_log.csv

    Args:
        query (str): The user query
        runtime (float): Seconds the pipeline took
        shortlist_size (int): Number of candidates returned
        blend_size (int): Number of items in current blend
        metrics (dict): Weighted blend metrics {Bitter, Cool, Sweet, Pungent, ...}
    """
    header = [
        "timestamp","query","runtime_sec","shortlist_size","blend_size",
        "Bitter","Cool","Sweet","Pungent","Umami","Astringent","Sour","Salty","Tasteless","TotalPct"
    ]

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "runtime_sec": round(runtime, 3),
        "shortlist_size": shortlist_size,
        "blend_size": blend_size,
        "Bitter": round(metrics.get("Bitter",0.0),3),
        "Cool": round(metrics.get("Cool",0.0),3),
        "Sweet": round(metrics.get("Sweet",0.0),3),
        "Pungent": round(metrics.get("Pungent",0.0),3),
        "Umami": round(metrics.get("Umami",0.0),3),
        "Astringent": round(metrics.get("Astringent",0.0),3),
        "Sour": round(metrics.get("Sour",0.0),3),
        "Salty": round(metrics.get("Salty",0.0),3),
        "Tasteless": round(metrics.get("Tasteless",0.0),3),
        "TotalPct": round(metrics.get("TotalPct",0.0),2),
    }

    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
