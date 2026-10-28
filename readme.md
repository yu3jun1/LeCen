# Lesion-Centric World Model Design

Lesion-Centric CT 世界模型第一轮核心实验代码：检测 D0a / D1 / D2，生存 S0–S5（S3 仅在治疗有变化时运行），机制检查（恢复梯度、打乱 h、信息丢失链）。
第一版只用固定划分与单一随机种子，不做多折与重复种子实验。

## 环境

Python 3.10；PyTorch 2.x；MONAI 1.x；SimpleITK、nibabel；pylidc；lungmask；scipy；pandas；OmegaConf；lifelines；pycox；scikit-survival；idc-index；matplotlib。

- pylidc 0.2.3 使用 `np.int` / `np.bool`，需要 numpy < 1.24。
- CT-CLIP：安装官方仓库中的 `transformer_maskgit`（提供 `CTViT`）。
- 每次运行会把 `pip freeze` 写入运行目录的 `env.lock`。

## 需要填写的接口

数据与权重路径都以 `???` 留空，使用前在 `configs/data.yaml` 中填写（或在命令行用 `key=value` 覆盖）：

| 配置项 | 内容 |
|---|---|
| `paths.raw_luna16` | `subset0.zip … subset9.zip` 所在目录 |
| `paths.luna16_seg` | seg-lungs-LUNA16 解压目录 |
| `paths.luna16_annotations` / `paths.luna16_excluded` | `annotations.csv` / `annotations_excluded.csv` |
| `paths.raw_clinical` | 临床 DICOM 下载目录 |
| `paths.clinical_series` | 资格核验确定的治疗前基线 CT：`patient_id, series_uid`（一患者一行） |
| `paths.clinical_table` | 原始临床表 |
| `paths.processed` / `paths.meta` / `paths.cache` | 输出目录 |
| `ctclip.checkpoint` / `ctclip.token_layer` | CT-CLIP 权重；CTViT 中输出 (1, 24, 24, 24, 512) token 网格的模块名 |
| `clinical.columns` / `clinical.event_values` / `clinical.censor_values` / `clinical.covariates` | 临床表列名映射；事件与删失的原始取值（其余取值视为结局未知） |
| `configs/det/base.yaml: adapt.iters` | 临床定位适配步数（仅在覆盖不足时） |
| `configs/clin/*.yaml: slots.extractor` | 提取器运行目录（S1/S2/S3/S5 用 D1，S4 用 D2） |

另需：`<meta>/clinical_tumors.csv`（`patient_id, tumor_id, center_world_xyz, bbox_world, source`，`bbox_world` 为世界坐标 cccwhd，向量列以空格分隔）；官方评测脚本放入 `eval/luna16_official/`；pylidc 的 DICOM 路径（`~/.pylidcrc`，只需 888 个序列）。

## 命令

```bash
# 数据
python -m data.crop_rules --cfg configs/data.yaml --fit-on train      # 确定外扩量并写回配置
python -m data.prep_luna16 --cfg configs/data.yaml --subset 0          # 逐个子集 0…9
python -m data.build_instances --cfg configs/data.yaml
python -m data.make_splits --cfg configs/data.yaml      # 同时确定临床合格队列：meta/clinical.csv、clinical_excluded.csv
python -m data.prep_clinical --cfg configs/data.yaml    # 只处理合格队列中的基线 CT
python -m data.crop_rules --cfg configs/data.yaml --audit             # 事后保留率审计
pytest tests/ -k "coords or sampling or lidc or ctclip or crop or eval_oracle or dataset"

# 检测
python -m tools.make_init --cfg configs/det/D1.yaml --seed 1000
python -m engine.train_det --cfg configs/det/D1.yaml                              # 单卡
python -m engine.train_det --cfg configs/det/D2.yaml
torchrun --nproc_per_node=<GPU 数> -m engine.train_det --cfg configs/det/D1.yaml   # DDP
python -m engine.infer_det --run runs/D1/<run_id> --split val          # 确定 p_min
python -m engine.infer_det --run runs/D1/<run_id> --split test --final
python -m eval.froc --run runs/D1/<run_id> --split test
python -m eval.compare --a runs/D1/<id> --b runs/D2/<id> --metrics cpm,sens1fp_3to6 --bootstrap 2000
python -m eval.shuffle_h --run runs/D1/<run_id>
python -m eval.recall_chain --run runs/D1/<run_id> --split val
# D0a：用 MONAI bundle 在 <meta>/monai_datalist_det.json 上从头训练与推理后
python -m engine.infer_det --run runs/D0a/<name> --split test --final --monai-json <MONAI 推理输出>

# 临床
python -m eval.recall_chain --run runs/D1/<run_id> --split val --cohort clinical
python -m engine.adapt_clinical --extractor runs/D1/<id> adapt.iters=<步数>   # 仅在覆盖不足时；也可用 torchrun 启动
python -m engine.extract_slots --extractor runs/D1/<id>
python -m engine.extract_slots --extractor runs/D2/<id>
python -m engine.train_surv --cfg configs/clin/S5.yaml slots.extractor=runs/D1/<id>
python -m eval.surv_eval --run runs/S5/<run_id> --final
python -m eval.compare --a runs/S5/<id> --b runs/S2/<id> --metrics cindex_ipcw --bootstrap 2000
```

## DDP

检测训练与临床定位适配可用 `torchrun` 启动（NCCL）。`train.batch` 是全局 batch，按卡均分，需能被卡数整除；
采样器在每个 rank 上生成同一条全局样本序列并按下标取模分配，所以任意卡数下每个全局 batch 的样本相同。
多卡时损失加入 0·Σp（`ddp.zero_loss_for_unused`），某个 rank 某步没有可恢复病灶或没有 GT 时参数仍在计算图中。
验证集按 rank 分片推理后汇总；日志、检查点与运行目录只由主进程写。D1 与 D2 须使用相同的卡数。
推理、评测、slot 提取与生存训练为单卡。

## 约定

- 数组轴序 (Z, Y, X)；物理坐标为 SimpleITK 世界坐标 (x, y, z)，单位 mm。
- 模型内部框：中心 (cz, cy, cx) + 尺寸 (dz, dy, dx)，单位为当前 patch 体素；与世界坐标、MONAI 框格式的转换只在 `common/boxes.py` 与 `common/coords.py`；所有 `grid_sample` 坐标只由 `common/sampling.py` 生成。
- 强度 x = (clip(HU, −1000, 400) + 1000) / 1400；HU 误差 = 1400 × 强度误差。
- GT 只用于训练监督、验证集选择和事后审计；测试集评测需显式 `--final`，并登记在 `runs/test_log.csv`，同一配置重复评测会被拒绝。
