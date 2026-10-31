# Lesion-Centric World Model Design

Lesion-Centric CT 世界模型第一轮核心实验代码：检测 D0a / D1 / D2，生存 S0–S5（S3 仅在治疗有变化时运行），机制检查（恢复梯度、打乱 h、信息丢失链）。
第一版只用固定划分与单一随机种子，不做多折与重复种子实验。

本轮队列已执行结束，结果与有效性审计见[最终实验报告](docs/reports/2026-10-31-final-experiment-report.md)。审计确认：D0a现存推理加载的bundle权重没有对应训练best165；S4/S5因配置访问冲突实际执行g而非预期f。原始产物保留，不能把这些名义配置的分数当作相应设计已被正确验证；本次报告没有修改实现或重跑实验。

轻量结果文件：[检测指标CSV](docs/reports/2026-10-31-final-results/detection_metrics.csv)、[生存指标CSV](docs/reports/2026-10-31-final-results/survival_metrics.csv)。完整训练日志、模型权重与冻结配置保留在本地原始 `runs/` 目录；运行路径和有效性说明见最终报告。

## 环境

Python 3.10；PyTorch 2.x；MONAI 1.x；SimpleITK、nibabel；pylidc；lungmask；scipy；pandas；OmegaConf；lifelines；pycox；scikit-survival；idc-index；matplotlib。

- pylidc 0.2.3 使用 `np.int` / `np.bool`，需要 numpy < 1.24。
- CT-CLIP：安装官方仓库中的 `transformer_maskgit`（提供 `CTViT`）。
- 每次运行会把 `pip freeze` 写入运行目录的 `env.lock`。

## 本机数据与权重路径

`configs/data.yaml` 已对齐 `/data/tanyuejun` 下的现有数据。主配置使用永久目录；
当前正式训练的 SSD 输入路径记录在各自实际运行目录的 `config.yaml`；目录索引见 `runs/index.json` 中。
命令行仍可用 `key=value` 覆盖路径。

| 配置项 | 内容 |
|---|---|
| `paths.raw_luna16` | `/data/tanyuejun/LUNA16/processed/LeCen/_scratch/raw_workspace`，含 10 个 ZIP 的软链接；原包在 `LUNA16/raw/archives` |
| `paths.luna16_seg` | `/data/tanyuejun/LUNA16/raw/seg_lungs/seg-lungs-LUNA16` |
| `paths.luna16_annotations` / `paths.luna16_excluded` | 已填入现有 `annotations.csv` / `annotations_excluded.csv` |
| `paths.luna16_evaluation_script` | `/home/tanyuejun/LeCen_runtime/luna16_official_py3/noduleCADEvaluationLUNA16.py` |
| `paths.processed` / `paths.meta` | 检测：`/data/tanyuejun/LUNA16/processed/LeCen`、`/data/tanyuejun/LUNA16/meta/LeCen` |
| `paths.clinical_meta` / `paths.clinical_processed` | `/data/tanyuejun/_shared/clinical/LeCen/lung1_external_v1_20261030/meta`；`/data/tanyuejun/_shared/clinical/LeCen/lung1_external_v1_20261030/processed`；新协议独立目录，保留旧 ACRIN 冻结版本 |
| `clinical.cohorts.lung1.raw` / `.prep` | `/data/tanyuejun/NSCLC-Radiomics/dicom`；`/data/tanyuejun/NSCLC-Radiomics/preparation/LeCen_lung1_external_v1_20261030` |
| `clinical.cohorts.acrin.raw` / `.frozen_meta` | `/data/tanyuejun/ACRIN-NSCLC-FDG-PET/dicom`；已冻结的115例模型输入 `verified_v1_20261029/model_inputs/meta` |
| `clinical.cohorts.acrin.prep` / `.bamf` | `/data/tanyuejun/ACRIN-NSCLC-FDG-PET/preparation/LeCen_lung1_external_v1_20261030`；`/data/tanyuejun/ACRIN-NSCLC-FDG-PET/annotations/BAMF-AIMI/dicom` |
| `paths.cache` | `/data/tanyuejun/_shared/cache/LeCen_lung1_external_v1_20261030`；新协议独立缓存，已有 LUNA tokens 通过链接复用 |
| `ctclip.checkpoint` | `/data/tanyuejun/_shared/weights/ctclip/CT-CLIP_v2.pt` |
| `ctclip.token_layer` | `""`，使用 encoder 本身，与现有 12³ token 缓存一致 |
| `configs/det/base.yaml: init_ckpt` | `/data/tanyuejun/_shared/checkpoints/LeCen/init.pt`，D1/D2 共用 |
| `configs/det/D0a_monai.yaml: bundle_root` | `/data/tanyuejun/_shared/weights/monai_bundle/lung_nodule_ct_detection` |
| `configs/clin/*.yaml: slots.extractor` | S2/S3/S5 由队列注入 D1 临床 slot 的实际运行目录，S4 注入 D2；手动运行需传 `slots.extractor`。S0、S1 不使用检测器 |

预处理版本为 `lecen_pp_v1_ctclip_v2_mean12_20261028`，LUNA16 外扩为 10 mm，均与现有缓存一致。
`~/.pylidcrc` 已指向 `/data/tanyuejun/LIDC-IDRI/dicom`。

### 临床队列：Lung1 开发，ACRIN 外部测试

- **开发队列 NSCLC-Radiomics（Lung1，MAASTRO，IDC `nsclc_radiomics`）**：422例NSCLC公开队列，主体接受根治性放疗或放化疗（官方保留的LUNG1-128是术后病例，无GTV-1）；
  OS 来自 `Survival.time` / `deadstatus.event`。按 (event, treatment) 分层 60/20/20，种子2026。
  主肿瘤框为 RTSTRUCT 中 GTV-1 的外接框（放疗科人工勾画），用于验证集覆盖率、slot 选择与临床定位适配。
  文献报告少数病例的 GTV 文件错误或与 CT 错位；`tumors_missing.csv` 记录没有 GTV-1 的病例，错位病例无法自动识别。
- **外部测试队列 ACRIN 6668**：沿用已冻结的115例与审计后的 OS（来源与核验见[临床核验报告](docs/reports/2026-10-29-acrin-clinical-verification.md)），
  不参与训练、调参、slot 选择或任何阈值确定，整体为 `split = external`，不需要人工肿瘤标注。
  BAMF-AIMI 的 PET/CT AI 分割（IDC `bamf_aimi_annotations`，少数经放射科或非专家修正）转为框，只用于报告外部队列主肿瘤覆盖率。
  这些分割定义在 PET 网格上：只采用与所选基线 CT 的 FrameOfReferenceUID 相同的 SEG，其余记入 `tumors_unmatched.csv`；AI 分割可能过大或含多处病灶。
- **统一变量**：age、sex_male、stage（I / II / IIIA / IIIB / other；ACRIN 的 IIB 记为 II）。ACRIN 的 Zubrod PS 在 Lung1 中没有，不进入模型。
  两个队列都没有可比的治疗细分，treatment 统一为 RT±chemo，S3 不运行。
  标准化、缺失填补、G0 标准化、生存时间分箱、t\*、τ 与删失分布都只在 Lung1 训练集上拟合，外部队列沿用。
- ACRIN 此前以 69/23/23 划分完成的 S0/S1 正式测试属于旧设计，保留记录；新设计下 ACRIN 的结果来自 Lung1 训练的模型。

本机下载依赖使用独立的 `/home/tanyuejun/LeCen_runtime/idc_venv`（idc-index 0.13.4 / IDC v25），以适配其pandas依赖；GPU预处理及训练仍使用原 `/home/tanyuejun/LeCen_runtime/venv`。
五步准备协调器与日志位于 `/data/tanyuejun/_management/LeCen/lung1_external_20261030/`，按顺序执行并在全部数据验证后接续新协议队列。S0/S1旧设计的测试登记已备份移除，旧完整运行结果保留。
路径、源数据核对及验证范围见[本轮准备记录](docs/reports/2026-10-30-lung1-acrin-preparation.md)。

本地元数据版本为 `lung1_external_v1_20261030`，固定种子2026：Lung1为训练253例、验证85例、内部测试84例；ACRIN的115例全部为外部测试，本轮无临床资格排除。Lung1已取得422个CT与422个RTSTRUCT，共51,652个DICOM文件，421例有GTV-1框，LUNG1-128保留为无GTV标注病例。LUNG1-035的33个未解析引用仅位于RT全局索引，全部8个实际ROI引用均可用；首次审计、官方目录核对及来源限制均保留在管理目录，不修改原DICOM。

ACRIN的BAMF准备记录为83例坐标系不匹配、28例无SEG、4例只有非肿瘤标签，可用框为0；115例仍全部参与外部生存测试，肿瘤覆盖率按无可用参考标注报告，不把这些病例视为无肿瘤。两个队列共537例的新临床体积、targets、crops与tokens已在上述独立版本目录中生成并核验；五步准备完成，新协议实验队列已自动接续。完成记录见管理目录的`pipeline_status.json`。

临床定位适配的共同预算为`configs/det/base.yaml`中的`adapt.iters=2000`。仅按Lung1验证集的既定覆盖规则触发，D1/D2相同，不使用测试分数选择步数。

## 命令

```bash
# 检测数据准备已完成；以下命令供重建时使用。临床按上述新协议准备。
python -m data.crop_rules --cfg configs/data.yaml --fit-on train      # 确定外扩量并写回配置
python -m data.prep_luna16 --cfg configs/data.yaml --subset 0          # 逐个子集 0…9
python -m data.build_instances --cfg configs/data.yaml
# python -m data.make_splits --cfg configs/data.yaml --part det    # 从零重建备用；当前检测划分不重跑
/home/tanyuejun/LeCen_runtime/idc_venv/bin/python -m data.prep_lung1 --cfg configs/data.yaml  # Lung1：CT + RTSTRUCT 与 GTV-1 框
/home/tanyuejun/LeCen_runtime/idc_venv/bin/python -m data.prep_acrin --cfg configs/data.yaml  # ACRIN：冻结115例与 BAMF 框
python -m data.make_splits --cfg configs/data.yaml --part clinical   # 合并队列：Lung1 划分，ACRIN = external
python -m data.prep_clinical --cfg configs/data.yaml --cohort lung1  # CT-CLIP token、肺裁剪、clinical_targets
python -m data.prep_clinical --cfg configs/data.yaml --cohort acrin
python -m data.crop_rules --cfg configs/data.yaml --audit             # 事后保留率审计
pytest tests/ -k "coords or sampling or lidc or ctclip or crop or eval_oracle or dataset"

# 检测（首轮 D1/D2 已完成30000步；以下为复现入口，共享初始化直接复用）
# 仅在明确需要重新生成共享初始化时执行，会写入 cfg.init_ckpt：
# python -m tools.make_init --cfg configs/det/D1.yaml --seed 1000
python -m engine.train_det --cfg configs/det/D1.yaml                              # 单卡
python -m engine.train_det --cfg configs/det/D2.yaml
torchrun --nproc_per_node=<GPU 数> -m engine.train_det --cfg configs/det/D1.yaml   # DDP
python -m engine.infer_det --run runs/D1/20261029-122606 --split val          # 训练完成后确定 p_min
python -m engine.infer_det --run runs/D1/20261029-122606 --split test --final
python -m eval.froc --run runs/D1/20261029-122606 --split test
python -m eval.compare --a runs/D1/20261029-122606 --b runs/D2/20261029-120947 --metrics cpm,sens1fp_3to6 --bootstrap 2000
python -m eval.shuffle_h --run runs/D1/<新 run_id>                 # 旧 D1 运行没有 recovery.crop_mm，不能用新目标评估
python -m eval.window_core --run runs/D1/20261029-122606         # 验证集：关闭 / 开启 window_core 对照，不改 p_min 与 preds
python -m eval.recall_chain --run runs/D1/20261029-122606 --split val
# D0a 当前目录为 runs/D0a/20261029-055704-resume-e134；MONAI 训练与推理由外部运行器和自动队列接续。
# 推理产物生成后，可手动转换 MONAI 输出（正式测试只登记一次）：
python -m engine.infer_det --run runs/D0a/20261029-055704-resume-e134 --split val --monai-json <val 推理输出> --monai-config <val 推理配置>
python -m engine.infer_det --run runs/D0a/20261029-055704-resume-e134 --split test --final --monai-json <test 推理输出> --monai-config <test 推理配置>

# 临床：以下为复现入口；当前正式任务由自动队列接续
# D1_CLINICAL_RUN、D2_CLINICAL_RUN、S5_RUN、S2_RUN 应填 runs/index.json 中已登记的实际路径
python -m engine.train_surv --cfg configs/clin/S0.yaml
python -m engine.train_surv --cfg configs/clin/S1.yaml
python -m eval.recall_chain --run ${D1_CLINICAL_RUN} --split val --cohort clinical
python -m eval.recall_chain --run ${D1_CLINICAL_RUN} --split external --cohort clinical   # ACRIN 覆盖率（BAMF 框，只报告）
python -m engine.adapt_clinical --extractor ${D1_CLINICAL_RUN} adapt.iters=<步数>   # 仅在覆盖不足时；也可用 torchrun 启动
python -m engine.extract_slots --extractor ${D1_CLINICAL_RUN}
python -m engine.extract_slots --extractor ${D2_CLINICAL_RUN}
python -m engine.train_surv --cfg configs/clin/S5.yaml slots.extractor="${D1_CLINICAL_RUN}"
python -m eval.surv_eval --run ${S5_RUN} --final          # Lung1 测试集与 ACRIN 外部队列各评一次
python -m eval.compare --a ${S5_RUN} --b ${S2_RUN} --metrics cindex_ipcw --bootstrap 2000
python -m eval.compare --a ${S5_RUN} --b ${S1_RUN} --metrics cindex_ipcw --bootstrap 2000 --split external
```

## 运行目录与自动接续

每个实验使用 `runs/<实验名>/<时间戳>/`，不建立 `current` 目录或别名；实际路径和状态登记在 `runs/index.json`。
异常结束且已废弃的旧运行已从 runs 清除，故障复现所需的日志、配置、源码和少量现场权重单独保存在 `/data/tanyuejun/_management/LeCen/runs_cleanup_20261029/`。
已完成的 D0a/D1/D2 及旧设计的 S0/S1 保留，新临床协议使用新的实际运行目录。D0a 以最后完整运行 `20261029-055704-resume-e134` 为准，共200epoch，验证集选中的 `checkpoints/model.pt` 为epoch165；`last_training.pt` 为epoch200状态。

自动队列每180秒检查数据与依赖，准备就绪后并行接续；只使用GPU1–7，优先2–7，GPU0禁用。
D1/D2完成后先分析训练更新、验证曲线及定位信息丢失链，再判断是否需要修复或延长训练。需要重训时由本会话自动接续；共同逻辑改动保持共享初始化及比较协议。
最终测试和依赖检测器的临床任务等待此次分析完成；测试集仅用于最终报告，不参与选择重训方案。
slot 选择与临床定位适配只使用 Lung1 的 GTV-1 框；ACRIN 不需要人工肿瘤标注，BAMF 框只用于报告覆盖率，不生成伪造真值。

本轮D1/D2均完成连续30000次有限更新；验证集CPM选择的最佳检查点分别为28000、20000步，完整三万步权重保留在各自`milestones/step_030000/`。
完成后的验证复核修复了float32概率与float64阈值比较的同分边界：`p_min`仍严格大于，LUNA工作点按FROC阈值包含等分候选，slot过滤使用同一严格比较函数。该修复不改变已训练的参数。
最佳与末步权重已各自重做89例验证；四者FP覆盖均达8 FP/scan，CPM与灵敏度排序不变，最佳权重的自实现与官方CPM一致。当前没有欠训或训练故障的确证，本轮保留30k预算并继续最终测试；不能以D1弱于D2为理由要求重训至结论反转。

## 生存配置

所有影像配置共用同一个全局向量 G0：缓存的 12×12×12 CT-CLIP token 在空间维度上的均值（512 维，按 Lung1 训练集标准化），
与检测器无关；它是 token 均值池化表示，不是 CT-CLIP 官方的图像级 latent。病灶 slot 为 256 维。

| 配置 | 输入 | 主要比较 |
|---|---|---|
| S0 | Cox：临床变量 + 治疗 one-hot | 影像是否带来临床之外的信息 |
| S1（G0） | `[g_global; c; e(a)]`，不使用检测器、slot 与 GlobalAdapter | 整幅 CT 编码基线 |
| S2 | D1 slot，先聚合再用 g 更新 | S5 vs S2：条件更新的位置 |
| S3 | S5 去掉治疗输入（仅在治疗有变化时运行；Lung1 + ACRIN 设计下不运行） | 治疗输入是否必要 |
| S4 | D2 slot，其余同 S5 | S5 vs S4：恢复训练对 slot 的作用 |
| S5 | D1 slot，逐病灶更新后聚合：`[r; m; g_global; c; e(a)]` | S5 vs S1：病灶中心表征 vs 整幅 CT 编码 |

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

检测训练、验证和临床定位适配直接使用 BF16 autocast；热图共享卷积也使用该路径。
不再使用 GradScaler、loss scale 或跳步逻辑，也不保留 FP16 配置回退。
梯度范数非有限时，在 AdamW 更新前报错并保存模型、优化器、调度器、梯度与当前 batch；不替换 NaN、不跳过样本。
D1 的恢复梯度检查使用未缩放的实际梯度。检测头已经输出 FP32 的 exist_logit 和 box，推理不重复转换概率类型。

首轮 D1/D2 均从相同 `init.pt` 开始：全局 batch 4、30000 次更新、每 2000 步验证，余弦学习率在第 30000 步衰减完成。
`pycox==0.3.0` 已安装在本机运行环境中，核心 PyTorch/NumPy/Pandas 版本保持不变。
跳步审计、精简方案及验证见 [AMP 缩放故障报告](docs/reports/2026-10-29-amp-scale-underflow.md)。
早期热图 FP16 失败证据保留在 [历史热图报告](docs/reports/2026-10-29-heatmap-overflow-fix.md)。

### 审计后的两项实现更正

- **生存模型更新位置**：配置键由 `model.update` 改为 `model.update_at`（`lesion` = f，S4 / S5；`pooled` = g，S2）。
  OmegaConf 的 DictConfig 是 MutableMapping，`cfg.update` 取到的是 dict 的方法而不是配置值，首轮 S4/S5 因此全部执行了 g。
  模型在构造时校验取值，forward 记录实际执行的分支；每次神经生存训练写出 `model_branch.json`（配置值与实际执行值），两者不一致时报错。
  `tests/test_update_branch.py` 检查：S2 为 pooled、S4/S5 为 lesion；S4 与 S5 的完整配置只差 `slots.extractor`；
  S5 与 S2 的模型配置只差 `update_at`；forward 实际走对应分支；同一组参数下两种结构的输出不同。
- **D0a 权重归属**：`infer_det --monai-json` 必须同时给出 MONAI 实际使用的推理配置 `--monai-config`。
  转换前按 bundle 表达式解析 `load_pretrain` 与 `checkpointloader#load_path`，要求启用加载且加载文件与 `<run>/checkpoints/model.pt`
  （验证集选出的 epoch 165）SHA256 相同，否则拒绝转换，也不登记测试；通过时写出 `monai_<split>_provenance.json`。
  首轮推理配置加载的是 bundle 自带 `models/model.pt`。外部运行器生成推理配置时须把 load_path 指向该运行的 checkpoint。
- 两项都需要重跑：S2 / S4 / S5 重新训练与评估（S2 的结构未变，重跑应与首轮 S2 预测一致，可作复现核对）；D0a 只需用正确权重重新推理验证集与测试集。
  首轮这些配置已在 `runs/test_log.csv` 登记，重新正式评测前须先决定如何处理旧登记。
  Lung1 test 与 ACRIN external 已经看过，这次重跑属于对预先设定设计的实现更正，不调整超参数或选择规则；确认性结论需要新的独立测试队列。

### D1 / D2 首轮结果后的两项修改

- **滑窗中心区域（`infer.window_core: true`）**：每个位置只保留中心离它最近的窗口的预测（相邻窗口中心的中点为界），
  热图候选与 decoder 输出都如此过滤。本轮 `overlap=0.25` 时，尾部补窗可让每轴出现三重覆盖，三维同一位置最多被 27 个窗口预测；
  NMS 取最高分，窗口边缘、上下文不完整的预测也参与排名。`eval.window_core` 在验证集上对同一权重比较关闭与开启，结果只写入 `mech/`。
  本轮旧权重验证对照中两组 CPM 均下降，因此共同关闭 core、保留 P4，见[验证决策与配对重训方案](docs/reports/2026-10-30-p0-p2-retraining.md)。
- **Recovery 目标改为固定物理尺寸（`model.recovery.crop_mm: 32`）**：以 GT 中心裁剪 32 mm 立方体，重采样为 32³（1 mm 体素）。
  原目标按 GT 框 × 1.5 缩放，每个病灶都是居中、占比固定的团块，平均模板即可重建（首轮 shuffle-h：Dice 0.760 vs 0.748），
  小结节只剩几个体素插值放大。新目标不再按 GT 框尺寸缩放，但仍有 GT 居中先验，且部分大病灶超出固定视野；不能仅由目标形式断言 h 已编码病灶特异信息。
- **shuffle-h 增加模板基线**：按逐体素有效性加权的目标平均（在验证评估集上取得，偏向模板）。true 需同时明显优于 shuffled 与 template；有效目标不足两个时拒绝比较。
- 信息丢失链另记各阶段的独立命中（`only_dec` 等），区分"热图候选落点偏离但 decoder 仍找到结节"与真正的漏检。
- 首轮 D1 / D2 的运行配置没有这两个字段：只能用 `eval.window_core` 做推理对照；其余推理、slot 提取和 shuffle-h 需要新运行。
  两项修改都影响训练期验证选权重，D1 与 D2 应从同一份共享初始化、以相同设置一起重训，比较协议不变。

可选 `train.failure_dir` 用于前向、反向及更新前检查失败的现场保存。
普通 `best.pt` / `last.pt` 仍只保存模型、recovery 和 step，不能用作完整状态续训。

- 数组轴序 (Z, Y, X)；物理坐标为 SimpleITK 世界坐标 (x, y, z)，单位 mm。
- 模型内部框：中心 (cz, cy, cx) + 尺寸 (dz, dy, dx)，单位为当前 patch 体素；与世界坐标、MONAI 框格式的转换只在 `common/boxes.py` 与 `common/coords.py`；所有 `grid_sample` 坐标只由 `common/sampling.py` 生成。
- 强度 x = (clip(HU, −1000, 400) + 1000) / 1400；HU 误差 = 1400 × 强度误差。
- GT 只用于训练监督、验证集选择和事后审计；测试集评测需显式 `--final`，并登记在 `runs/test_log.csv`，同一配置重复评测会被拒绝。
