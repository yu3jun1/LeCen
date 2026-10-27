"""nii.gz / npz / csv / json 读写，配置加载，运行目录与测试集登记。"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from omegaconf import OmegaConf

from common.hashing import file_sha1

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    """加载 YAML；`defaults` 列出的父配置（相对本文件）先合并，再合并本文件与命令行 dotlist。"""
    path = Path(path)
    cfg = OmegaConf.load(path)
    parents = [load_config(path.parent / p) for p in cfg.pop("defaults", [])]
    cfg = OmegaConf.merge(*parents, cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.name = path.stem
    return cfg


def read_image(path):
    return sitk.ReadImage(str(path))


def read_header(path):
    """只读图像头（origin / spacing / direction / size），接口与 sitk.Image 相同。"""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r


def write_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_npz(path, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def write_csv(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_csv(path, row):
    path = Path(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def upsert_csv(path, df, key):
    """用 df 替换 csv 中 key 相同的行（逐子集预处理时更新同一张表）。"""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, dtype={key: str})
        df = pd.concat([old[~old[key].isin(df[key])], df], ignore_index=True)
    write_csv(path, df)


def world_list(s):
    """csv 中以空格分隔保存的向量列 → (n, k) 数组。"""
    return np.array([np.array(v.split(), dtype=float) for v in s])


def vec_str(v):
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def make_run(cfg, hashes):
    """runs/<config_name>/<run_id>/：写入合并后的配置、env.lock、种子与哈希。"""
    run_dir = ROOT / "runs" / cfg.name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    (run_dir / "env.lock").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    write_json(run_dir / "seeds.json", OmegaConf.to_container(cfg.seeds) if "seeds" in cfg else {"seed": cfg.seed})
    write_json(run_dir / "hashes.json", {"git_commit": git_commit(), **hashes})
    return run_dir


def register_final(config_name, run_dir, ckpt):
    """测试集只评一次：在 runs/test_log.csv 中登记，同一配置重复评测被拒绝。"""
    log = ROOT / "runs" / "test_log.csv"
    df = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["config", "run", "checkpoint", "time"])
    if config_name in set(df["config"]):
        raise RuntimeError(f"{config_name} 已在测试集上评测过：{log}")
    row = {"config": config_name, "run": str(run_dir), "checkpoint": str(ckpt), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_csv(log, pd.concat([df, pd.DataFrame([row])], ignore_index=True))


def data_hashes(cfg):
    meta = Path(cfg.paths.meta)
    return {
        "splits_det": file_sha1(meta / "splits_det.csv"),
        "splits_clin": file_sha1(meta / "splits_clin.csv"),
        "crop_luna_margin_mm": cfg.preprocess.crop.luna.margin_mm,
        "crop_clinical_margin_mm": cfg.preprocess.crop.clinical.margin_mm,
        "preprocessing_version": cfg.preprocess.version,
    }
