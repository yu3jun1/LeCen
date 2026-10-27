"""文件哈希、配置哈希、extractor_hash。"""
import hashlib

from omegaconf import OmegaConf


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg):
    return hashlib.sha1(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()


def extractor_hash(cfg, ckpt_path):
    """sha1(合并后的模型配置 + checkpoint 文件哈希 + preprocessing_version + ctclip 版本) 前 12 位。"""
    parts = [
        config_hash(cfg.model),
        file_sha1(ckpt_path),
        cfg.preprocess.version,
        f"{cfg.ctclip.checkpoint}:{cfg.ctclip.token_layer}",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def sel_hash(extractor_hash, sel_cfg):
    return hashlib.sha1(f"{extractor_hash}|{config_hash(sel_cfg)}".encode()).hexdigest()[:12]
