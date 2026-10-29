"""文件哈希、配置哈希、extractor_hash。"""
import hashlib

from omegaconf import OmegaConf


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_sha1(paths):
    """按顺序组合多个文件的 sha1。"""
    return hashlib.sha1("|".join(file_sha1(p) for p in paths).encode()).hexdigest()


def config_hash(cfg):
    return hashlib.sha1(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()


EXTRACTOR_KEYS = ["model", "patch_zyx", "infer", "preprocess", "ctclip"]


def extractor_hash(cfg, ckpt_path, p_min):
    """sha1(合并后影响提取的配置（模型、patch、推理、完整预处理、CT-CLIP）+ checkpoint 文件哈希 + p_min) 前 12 位。"""
    parts = [config_hash(OmegaConf.masked_copy(cfg, EXTRACTOR_KEYS)), file_sha1(ckpt_path), repr(p_min)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def sel_hash(extractor_hash, sel_cfg):
    return hashlib.sha1(f"{extractor_hash}|{config_hash(sel_cfg)}".encode()).hexdigest()[:12]
