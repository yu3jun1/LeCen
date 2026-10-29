# Lesion-Centric World Model Design

Lesion-Centric CT 世界模型第一轮核心实验代码：检测 D0a / D1 / D2，生存 S0–S5（S3 仅在治疗有变化时运行），机制检查（恢复梯度、打乱 h、信息丢失链）。
第一版只用固定划分与单一随机种子，不做多折与重复种子实验。

## 环境

Python 3.10；PyTorch 2.x；MONAI 1.x；SimpleITK、nibabel；pylidc；lungmask；scipy；pandas；OmegaConf；lifelines；pycox；scikit-survival；idc-index；matplotlib。

- pylidc 0.2.3 使用 `np.int` / `np.bool`，需要 numpy < 1.24。
- CT-CLIP：安装官方仓库中的 `transformer_maskgit`（提供 `CTViT`）。
- 每次运行会把 `pip freeze` 写入运行目录的 `env.lock`。

## 本机数据与权重路径

`configs/data.yaml` 已对齐 `/data/tanyuejun` 下的现有数据。主配置使用永久目录；
当前正式训练的 SSD 输入路径记录在各自 `runs/<D>/current/config.yaml` 中。
命令行仍可用 `key=value` 覆盖路径。

| 配置项 | 内容 |
|---|---|
| `paths.raw_luna16` | `/data/tanyuejun/LUNA16/processed/LeCen/_scratch/raw_workspace`，含 10 个 ZIP 的软链接；原包在 `LUNA16/raw/archives` |
| `paths.luna16_seg` | `/data/tanyuejun/LUNA16/raw/seg_lungs/seg-lungs-LUNA16` |
| `paths.luna16_annotations` / `paths.luna16_excluded` | 已填入现有 `annotations.csv` / `annotations_excluded.csv` |
| `paths.luna16_evaluation_script` | `/home/tanyuejun/LeCen_runtime/luna16_official_py3/noduleCADEvaluationLUNA16.py` |
| `paths.raw_clinical` | `/data/tanyuejun/ACRIN-NSCLC-FDG-PET/dicom`，支持患者／检查／序列目录 |
| `paths.clinical_series` | `null`：尚无已核验的治疗前基线清单；候选表单独登记在 `paths.clinical_series_candidates` |
| `paths.clinical_table` | `/data/tanyuejun/ACRIN-NSCLC-FDG-PET/preparation/meta/clinical_acrin.csv`，临床原始字段汇总，结局尚未审定 |
| `paths.processed` | `/data/tanyuejun/LUNA16/processed/LeCen` |
| `paths.meta` | `/data/tanyuejun/LUNA16/meta/LeCen` |
| `paths.cache` | `/data/tanyuejun/_shared/cache/LeCen` |
| `ctclip.checkpoint` | `/data/tanyuejun/_shared/weights/ctclip/CT-CLIP_v2.pt` |
| `ctclip.token_layer` | `""`，使用 encoder 本身，与现有 12³ token 缓存一致 |
| `configs/det/base.yaml: init_ckpt` | `/data/tanyuejun/_shared/checkpoints/LeCen/init.pt`，D1/D2 共用 |
| `configs/det/D0a_monai.yaml: bundle_root` | `/data/tanyuejun/_shared/weights/monai_bundle/lung_nodule_ct_detection` |
| `configs/clin/*.yaml: slots.extractor` | S1/S2/S3/S5：`/home/tanyuejun/LeCen/runs/D1_clinical/current`；S4：`D2_clinical/current`；由临床队列后续生成 |

预处理版本为 `lecen_pp_v1_ctclip_v2_mean12_20261028`，LUNA16 外扩为 10 mm，均与现有缓存一致。
`~/.pylidcrc` 已指向 `/data/tanyuejun/LIDC-IDRI/dicom`。

仍待核验的是临床语义配置：`clinical.columns` 中的结局／治疗列、`event_values`、`censor_values`、
`covariates`，以及仅在覆盖不足时使用的 `adapt.iters`。这些不是可凭目录补全的路径。
`<meta>/clinical_tumors.csv` 和 `splits_clin.csv` 当前仅有表头；临床提取器目录及 `slots.json` 尚未生成。
候选 CT 清单不能直接当作合格队列，S0–S5 需待这些材料就绪后运行。

## 命令

```bash
# 数据准备已完成；以下命令供重建时使用。临床相关命令需先完成上述核验。
python -m data.crop_rules --cfg configs/data.yaml --fit-on train      # 确定外扩量并写回配置
python -m data.prep_luna16 --cfg configs/data.yaml --subset 0          # 逐个子集 0…9
python -m data.build_instances --cfg configs/data.yaml
python -m data.make_splits --cfg configs/data.yaml      # 同时确定临床合格队列：meta/clinical.csv、clinical_excluded.csv
python -m data.prep_clinical --cfg configs/data.yaml    # 只处理合格队列中的基线 CT
python -m data.crop_rules --cfg configs/data.yaml --audit             # 事后保留率审计
pytest tests/ -k "coords or sampling or lidc or ctclip or crop or eval_oracle or dataset"

# 检测（当前 D1/D2 已在运行；初始化文件已存在，直接复用）
# 仅在明确需要重新生成共享初始化时执行，会写入 cfg.init_ckpt：
# python -m tools.make_init --cfg configs/det/D1.yaml --seed 1000
python -m engine.train_det --cfg configs/det/D1.yaml                              # 单卡
python -m engine.train_det --cfg configs/det/D2.yaml
torchrun --nproc_per_node=<GPU 数> -m engine.train_det --cfg configs/det/D1.yaml   # DDP
python -m engine.infer_det --run runs/D1/current --split val          # 训练完成后确定 p_min
python -m engine.infer_det --run runs/D1/current --split test --final
python -m eval.froc --run runs/D1/current --split test
python -m eval.compare --a runs/D1/current --b runs/D2/current --metrics cpm,sens1fp_3to6 --bootstrap 2000
python -m eval.shuffle_h --run runs/D1/current
python -m eval.recall_chain --run runs/D1/current --split val
# D0a 当前目录为 runs/D0a/current；MONAI 训练与推理由外部运行器和自动队列接续。
# 推理产物生成后，可手动转换 MONAI 输出（正式测试只登记一次）：
python -m engine.infer_det --run runs/D0a/current --split test --final --monai-json runs/D0a/current/monai_test_project_format.json

# 临床
python -m eval.recall_chain --run runs/D1_clinical/current --split val --cohort clinical
python -m engine.adapt_clinical --extractor runs/D1_clinical/current adapt.iters=<步数>   # 仅在覆盖不足时；也可用 torchrun 启动
python -m engine.extract_slots --extractor runs/D1_clinical/current
python -m engine.extract_slots --extractor runs/D2_clinical/current
python -m engine.train_surv --cfg configs/clin/S5.yaml
python -m eval.surv_eval --run runs/S5/current --final
python -m eval.compare --a runs/S5/current --b runs/S2/current --metrics cindex_ipcw --bootstrap 2000
```

## DDP

检测训练与临床定位适配可用 `torchrun` 启动（NCCL）。`train.batch` 是全局 batch，按卡均分，需能被卡数整除；
采样器在每个 rank 上生成同一条全局样本序列并按下标取模分配，所以任意卡数下每个全局 batch 的样本相同。
多卡时损失加入 0·Σp（`ddp.zero_loss_for_unused`），某个 rank 某步没有可恢复病灶或没有 GT 时参数仍在计算图中。
验证集按 rank 分片推理后汇总；日志、检查点与运行目录只由主进程写。D1 与 D2 须使用相同的卡数。
推理、评测、slot 提取与生存训练为单卡。

## 约定

检测器的参考框与逐层细化框将尺寸限制在每轴 `[1, 2 × patch]` 体素，防止无约束指数导致体积溢出。
decoder 的三处注意力使用 `need_weights=False`。真实 D1/D2 失败现场已确认：第一层自注意力的
FP16 分数先溢出，随后 NaN 传播到框与匹配矩阵；仅限制尺寸不足以修复这两次失败。
复现和分项验证见 [数值修复验证报告](docs/reports/2026-10-28-numerical-fix-verification.md)。

热图头的共享 `Conv3d → GroupNorm → ReLU` 使用 FP32，之后的分类和尺寸输出头继续使用 AMP。
D1 step 15441 与 D2 step 10457 的现场确认该卷积先超过 FP16 上限，再由 GroupNorm 传播 NaN；
复现、回归与多卡验证见 [热图卷积溢出修复报告](docs/reports/2026-10-29-heatmap-overflow-fix.md)。

可选 `train.failure_dir` 用于保存前向失败时的模型、优化器、scaler 和当前 batch。
该现场保存功能不提供自动恢复入口；普通 `best.pt` / `last.pt` 仍只保存模型、recovery 和 step。

- 数组轴序 (Z, Y, X)；物理坐标为 SimpleITK 世界坐标 (x, y, z)，单位 mm。
- 模型内部框：中心 (cz, cy, cx) + 尺寸 (dz, dy, dx)，单位为当前 patch 体素；与世界坐标、MONAI 框格式的转换只在 `common/boxes.py` 与 `common/coords.py`；所有 `grid_sample` 坐标只由 `common/sampling.py` 生成。
- 强度 x = (clip(HU, −1000, 400) + 1000) / 1400；HU 误差 = 1400 × 强度误差。
- GT 只用于训练监督、验证集选择和事后审计；测试集评测需显式 `--final`，并登记在 `runs/test_log.csv`，同一配置重复评测会被拒绝。
