"""生成共享模块初始权重 init.pt（含哈希）；D1、D2 都加载这一份。

python -m tools.make_init --cfg configs/det/D1.yaml --seed 1000
"""
import argparse
from pathlib import Path

import torch

from common.hashing import file_sha1
from common.io import load_config, write_json
from common.seed import seeded
from models.detector import Detector

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    with seeded(args.seed):
        model = Detector(cfg)
    path = Path(cfg.init_ckpt)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    write_json(path.with_suffix(".json"), {"seed": args.seed, "sha1": file_sha1(path), "cfg": args.cfg})
    print(path, file_sha1(path))
