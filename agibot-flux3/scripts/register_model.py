#!/usr/bin/env python3
"""Add a model to configs/models.json if its tag is not there yet.
    scripts/register_model.py <tag> <step> <label> <trained depths "1,2,4"> <config> <ema>
"""
import json, sys
from pathlib import Path

tag, step, label, trained, cfg, ema = sys.argv[1:7]
p = Path(__file__).resolve().parents[1] / "configs" / "models.json"
d = json.loads(p.read_text())
if any(m["tag"] == tag for m in d["models"]):
    print("registered already:", tag); sys.exit(0)
d["models"].append({"tag": tag, "step": int(step), "label": label, "trained_depths": [int(x) for x in trained.split(",") if x],
                    "config": cfg, "ema": ema, "note": ""})
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("registered", tag)
