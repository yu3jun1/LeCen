# Lung1 开发队列与 ACRIN 外部测试准备

基于 `main` 的 `55df258`，将临床输入迁移至独立版本 `lung1_external_v1_20261030`。旧 ACRIN 冻结元数据、缓存和完整实验结果保留。检测输入继续使用 LUNA16；临床输入不再借用检测运行中保存的旧路径。

## 数据与固定划分

| 队列 | 合格人数 | 训练 | 验证 | 内部测试 | 外部测试 |
|---|---:|---:|---:|---:|---:|
| NSCLC-Radiomics / Lung1 | 422 | 253 | 85 | 84 | 0 |
| ACRIN | 115 | 0 | 0 | 0 | 115 |

种子2026，按 `(event, treatment)` 分层；本轮无临床资格排除。协变量标准化、缺失填补、G0 标准化、时间分箱、时间范围和删失估计只使用 Lung1 训练集。ACRIN 保留此前冻结的115例与结局，不因缺少肿瘤参考标注而排除患者。

## 本地路径

| 配置 | 路径 |
|---|---|
| `paths.clinical_meta` | `/data/tanyuejun/_shared/clinical/LeCen/lung1_external_v1_20261030/meta` |
| `paths.clinical_processed` | `/data/tanyuejun/_shared/clinical/LeCen/lung1_external_v1_20261030/processed` |
| `clinical.cohorts.lung1.raw` | `/data/tanyuejun/NSCLC-Radiomics/dicom` |
| `clinical.cohorts.lung1.prep` | `/data/tanyuejun/NSCLC-Radiomics/preparation/LeCen_lung1_external_v1_20261030` |
| `clinical.cohorts.acrin.prep` | `/data/tanyuejun/ACRIN-NSCLC-FDG-PET/preparation/LeCen_lung1_external_v1_20261030` |
| `clinical.cohorts.acrin.bamf` | `/data/tanyuejun/ACRIN-NSCLC-FDG-PET/annotations/BAMF-AIMI/dicom` |
| `paths.cache` | `/data/tanyuejun/_shared/cache/LeCen_lung1_external_v1_20261030` |

`meta` 保存合并临床表、划分、肿瘤框及 `crops.csv`；`processed/clinical/` 保存患者 NIfTI，`processed/clinical_targets/` 保存目标 JSON；缓存下的 `ctclip/` 保存每患者 tokens。新缓存链接复用888份 LUNA tokens，临床 tokens 重新生成。

## 原始资源核验与来源限制

- 从 [TCIA NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) 的 IDC 镜像取得全部422个CT、422个RTSTRUCT，共51,652个DICOM文件。与TCIA官方NBIA全集目录逐序列比较：患者、Study UID、Modality及文件数无差异。
- 421例有 GTV-1 框；LUNG1-128无 GTV-1，保留患者与缺标注记录，不生成虚假肿瘤框。
- LUNG1-035的94张CT与TCIA、IDC一致。全部8个实际ROI的轮廓引用可解析；33个未解析SOP只存在于未被实际轮廓使用的全局索引。作为来源限制保留，不能宣称全局RT引用完全合规。首次失败报告、独立复核和修订后的全量审计都保留。实际轮廓引用缺失、身份或坐标系不符仍阻断准备。
- 原始层距/重复平面等几何备注保留在审计的逐序列记录中，不修改原DICOM，不据此自动删除患者。
- ACRIN BAMF可用框为0：83例坐标系不符、28例无SEG、4例没有肿瘤标签。外部肿瘤覆盖率应报告无可用参考标注，不能当作0灵敏度或无肿瘤。这不阻断115例的外部生存评测。

## 实现与验证

`extract_slots`、`adapt_clinical`、`recall_chain` 的临床元数据、体积与缓存读取当前配置，保留检测器的 LUNA 路径。`prep_clinical` 使用已下载的 lungmask 权重。空肿瘤表可正常生成空 targets 和缺标注覆盖率报告；BAMF选择只在肿瘤标签非空且FoR严格一致的对象间比较来源优先级。

`prep_lung1` 同时归档 IDC 下载清单和原始临床表。共同临床定位适配预算固定为`configs/det/base.yaml`中的`adapt.iters=2000`；仅在代码既定的Lung1验证覆盖门槛触发时运行，D1/D2使用同一值。门槛与启动程序统一读取该配置，避免从不含训练字段的`data.yaml`误读预算。该值在查看新定位覆盖结果前固定，不根据测试分数选择；相关43项CPU回归通过。

下载使用独立 `idc_venv`（idc-index 0.13.4 / IDC v25），不升级训练环境的 numpy/pandas。准备按 `prep_lung1 → prep_acrin → make_splits --part clinical → prep_clinical lung1 → prep_clinical acrin` 接续；每队列内部按患者分片调用正式预处理入口。

验证包括218项相关CPU回归、18项源审计测试、13项协调恢复测试和9项预处理恢复测试。真实035引用复核与TCIA全集交叉核对通过。GPU0禁止；预处理使用物理GPU2、4、5、7和1。

Lung1训练集的253例实际产物已单独核验：252例有GTV，252个目标框全部完整保留，世界坐标往返检查无差异；1例无GTV保留。G0为253×512，全部有限，512个通道均非恒定。未依据验证/测试结果调参，未修改裁剪设置。记录位于管理目录的`clinical_train_preprocessing_audit.json`和同名CSV。

原始CT以已审计目录及完整文件属性清单监测普通变更；模型、元数据、代码和输出另有内容SHA。文件属性指纹不等同原始DICOM内容哈希。每患者只有CT、targets、tokens及真实裁剪记录全部验证后才写完成记录，失败病例不跳过。

## 运行记录

管理目录：`/data/tanyuejun/_management/LeCen/lung1_external_20261030/`。

- 实时步骤与完成标志：`pipeline_status.json`。
- TCIA/IDC比较：`tcia_idc_crosscheck.json`；源核验：`lung1_source_audit.json`。
- 035独立复核：`lung1_035_reference_review.json`；原失败报告：`lung1_source_audit_initial_failed.json`。
- 每患者完成记录与分片日志：`preprocessing/<cohort>/`。
- 旧S0/S1测试登记备份：`test_log_before_lung1_external.csv`、`removed_legacy_test_registrations.json`。仅删除旧登记，旧运行和结果保留。

五步影像准备已全部完成：422例Lung1、115例ACRIN，各有CT、targets、tokens，合并`crops.csv`为537行；旧ACRIN冻结元数据6个文件的内容SHA保持不变。完整生产数据加载得到G0 `[537,512]`、临床变量`[537,7]`，全部有限；训练集标准化正常，四种split各一例的S1 CPU前向输出`[4,10]`且有限，未执行优化器更新。

新协议实验队列已自动启动，保留28项已完成任务并接续35项新任务。新模型的内部测试与外部测试分开写结果，测试分数不用于训练、适配或门槛选择。实时状态以管理目录和`/data/tanyuejun/_management/LeCen/experiment_queue/status.json`为准。
