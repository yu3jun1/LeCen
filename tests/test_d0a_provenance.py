"""D0a 预测转换前核对 MONAI 推理配置实际加载的权重：首轮推理加载了 bundle 自带权重而不是训练选出的 epoch。"""
import json

import pytest

pytest.importorskip("monai")

from engine.infer_det import verify_monai_checkpoint  # noqa: E402


def setup(tmp_path, load_path, load_pretrain=True):
    run, bundle = tmp_path / "run", tmp_path / "bundle"
    (run / "checkpoints").mkdir(parents=True)
    (bundle / "models").mkdir(parents=True)
    (run / "checkpoints" / "model.pt").write_bytes(b"trained epoch 165")
    (bundle / "models" / "model.pt").write_bytes(b"official bundle weights")
    config = tmp_path / "monai_infer_test_config.json"
    config.write_text(json.dumps({"bundle_root": str(bundle), "ckpt_dir": str(run / "checkpoints"), "load_pretrain": load_pretrain,
                                  "checkpointloader": {"_target_": "CheckpointLoader", "load_path": load_path,
                                                       "load_dict": {"model": "@network"}}}))
    return run, config


def test_bundle_weights_are_rejected(tmp_path):
    run, config = setup(tmp_path, "$@bundle_root + '/models/model.pt'")
    with pytest.raises(ValueError):
        verify_monai_checkpoint(run, config, "test")
    assert not (run / "monai_test_provenance.json").exists()


def test_disabled_loading_is_rejected(tmp_path):
    run, config = setup(tmp_path, "$@ckpt_dir + '/model.pt'", load_pretrain=False)
    with pytest.raises(ValueError):
        verify_monai_checkpoint(run, config, "test")


def test_trained_checkpoint_passes_and_is_recorded(tmp_path):
    run, config = setup(tmp_path, "$@ckpt_dir + '/model.pt'")
    record = verify_monai_checkpoint(run, config, "val")
    saved = json.loads((run / "monai_val_provenance.json").read_text())
    assert saved == record and record["loaded"] == str(run / "checkpoints" / "model.pt")
    assert record["loaded_sha256"] == record["expected_sha256"]
