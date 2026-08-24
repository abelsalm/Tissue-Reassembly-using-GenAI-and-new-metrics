#!/usr/bin/env python3
"""Print compact best-val summary from a domain-run history.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(history_path: Path) -> dict:
    hist = json.loads(history_path.read_text())
    best = None
    last = None
    for h in hist:
        last = h
        v = h.get("val")
        if not v:
            continue
        rec = {
            "epoch": h["epoch"],
            "val_acc": v["acc"],
            "val_loss": v["loss"],
            "train_acc": h["train"]["acc"],
            "train_loss": h["train"]["loss"],
            "lr": h.get("lr"),
        }
        if best is None or rec["val_acc"] > best["val_acc"]:
            best = rec
    return {
        "n_epochs": None if last is None else last["epoch"],
        "best": best,
        "last_train_acc": None if last is None else last["train"]["acc"],
        "path": str(history_path),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: summarize_domain_run.py <history.json or log run dir>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if p.is_dir():
        p = p / "history.json"
    print(json.dumps(summarize(p), indent=2))


if __name__ == "__main__":
    main()
