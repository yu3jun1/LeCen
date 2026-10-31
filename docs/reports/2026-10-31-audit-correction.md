# 2026-10-31 审计后实现更正：最终结果

完成核验时间：北京时间 2026-10-31 17:11:13。D0a正确权重推理、S2/S4/S5重训及原协议配对比较均已完成。

本记录接续[首轮产物审计](2026-10-31-final-experiment-report.md)。此前测试集已见，本轮是冻结原设计下的描述性实现更正，不是新的独立确认性试验；没有根据测试分数改变超参数、预算、划分或验证选择规则。

## 更正内容与真实性

- D0a加载原200epoch训练中按验证val_coco选出的epoch165：`runs/D0a/20261029-055704-resume-e134/checkpoints/model.pt`，SHA256为`4cd4091a6b55178c7894ce74f5c1d14c477f5c22077838e998eebe4eb3b123c0`；没有重新训练D0a。
- 实际加载前校验来源，推理前后核对权重及配置；生成receipt绑定配置、权重、原始/转换JSON及预测CSV。转换使用`--monai-json --monai-config`，测试追加`--final`；原运行`config.yaml`及名字保持冻结。
- D0a最终登记与临床评测共用原登记锁。队列调用已补齐第五参数，D0a不再提前登记或直接绕过校验转换。
- S2/S4/S5使用最新`model.update_at`，沿原CPU4线程、seed1000、数据release和冻结slots；三组均重新训练并保存实际分支记录。D1/D2及S0/S1复用原完成结果。
- 项目定向回归14项、运行器定向回归10项及队列调用签名检查通过。独立只读复核确认89/176扫描名单完整、两组bootstrap各2000、全部receipt绑定哈希一致。官方/自实现最大差分别为2.43e-10与4.98e-10。

## 检测：验证与最终测试

固定LUNA16划分：训练623、验证89、测试176扫描；验证111、测试223个结节。测试3–6/6–10/≥10mm为90/68/65。下面均为对应已训练模型的有效结果；测试沿用验证确定的权重与规则。

| 实验 | 选择权重 | 验证CPM | 最终test CPM（95% CI） | test sens@0.125FP | test sens@1FP | test 3–6mm sens@1FP |
|---|---|---:|---:|---:|---:|---:|
| D0a | epoch165 | 0.906049 | 0.909033 [0.860582, 0.945643] | 0.766816 | 0.923767 | 0.900000 |
| D1 | step58000 | 0.859717 | 0.839846 [0.776294, 0.890004] | 0.690583 | 0.869955 | 0.766667 |
| D2 | step52000 | 0.854569 | 0.796925 [0.744359, 0.848106] | 0.556054 | 0.860987 | 0.766667 |

正确D0a的test总体为206/223（92.38%），小结节为81/90（90.00%）。旧bundle结果0.937860已归档，不能继续作为训练best165的分数；当前正确模型test CPM为0.909033。验证CPM为0.906049，95% CI=[0.861702,0.944943]。

原协议检测配对比较（方向D1/D2−正确D0a；患者bootstrap2000、seed0）：

| 对比 | 指标 | 点估计差 | bootstrap差值均值 | 配对95% CI |
|---|---|---:|---:|---|
| D1−D0a | cpm | -0.069186 | -0.071763 | [-0.111445, -0.032734] |
| D1−D0a | sens1fp_3to6 | -0.133333 | -0.130248 | [-0.218393, -0.054047] |
| D2−D0a | cpm | -0.112108 | -0.108737 | [-0.152004, -0.064359] |
| D2−D0a | sens1fp_3to6 | -0.133333 | -0.128901 | [-0.215692, -0.048178] |

当前固定模型下D0a检测更高；但其架构、200epoch/global batch16/world4训练与D1/D2的60k/global batch4/world1不同，差异不能归因于单一模块。区间只反映固定模型的患者抽样不确定性。D1−D2原test CPM点差+0.042921，配对区间[-0.000161,+0.074838]仍跨0。没有据这些最终测试结果延长或重新训练。

## 生存：Lung1 test与ACRIN external

Lung1 train/val/test=253/85/84，ACRIN external=115。C为IPCW C-index，IBS越低越好；t*=tau=730天，IBS为原固定50点10–730天，删失分布等预处理只在Lung1 train拟合。每组患者bootstrap2000次。

| 实验 | 实际模型 | Lung1 test C（95% CI） | test IBS | ACRIN external C（95% CI） | external IBS |
|---|---|---:|---:|---:|---:|
| S0 | Cox临床基线（复用） | 0.503783 [0.420969, 0.588054] | 0.199262 | 0.566699 [0.483502, 0.652459] | 0.174218 |
| S1 | G0全局CT（复用） | 0.586097 [0.509361, 0.667774] | 0.198932 | 0.468578 [0.394840, 0.544257] | 0.275313 |
| S2 | pooled/g，D1 slots | 0.494193 [0.413377, 0.573745] | 0.199447 | 0.489448 [0.411374, 0.566388] | 0.186318 |
| S4 | lesion/f，D2 slots | 0.512737 [0.434731, 0.591355] | 0.191349 | 0.484620 [0.406427, 0.562664] | 0.197577 |
| S5 | lesion/f，D1 slots | 0.500966 [0.421622, 0.579544] | 0.192578 | 0.498217 [0.418024, 0.573504] | 0.187126 |
| S3 | 治疗类别恒定，不适用 | NA | NA | NA | NA |

三组新神经模型均完成23个epoch（编号0–22），验证损失最低在编号2（人读第3轮），随后按原patience20正常早停。`model_branch.json`均确认配置与执行一致：S2=pooled，S4/S5=lesion。

**S2精确复现通过**：17个权重张量、val/test/external全部18个预测数组字段、完整训练CSV数值与字节、种子及非Git来源哈希均与旧S2相同。身份核验未读取指标作选择。旧S4/S5是g模型，新结果来自真正lesion/f重训，不能混用。

原协议8项S5配对比较已完成（方向新S5−对照，2000次、seed0）：

| 对照 | 划分 | C点差 | 配对95% CI |
|---|---|---:|---|
| S0 | test | -0.002817 | [-0.106799, +0.103188] |
| S0 | external | -0.068482 | [-0.171636, +0.034267] |
| S1 | test | -0.085131 | [-0.194199, +0.017662] |
| S1 | external | +0.029639 | [-0.028895, +0.086764] |
| S2 | test | +0.006773 | [-0.027522, +0.042080] |
| S2 | external | +0.008768 | [-0.029002, +0.049469] |
| S4 | test | -0.011771 | [-0.066281, +0.042720] |
| S4 | external | +0.013596 | [-0.027048, +0.055038] |

8个区间均包含0，当前不能确认临床增益；正确f模型不再与S2重复。ACRIN仍缺可匹配的CT肿瘤参考框，定位灵敏度为NA，不是0；全部115人保留在外部生存评估中。固定单划分、单随机种子、小临床队列与域差异限制外推；这些已见测试的更正结果不能当作新的独立确认性验证。

## 登记、历史与运行路径

完整原测试登记已备份；旧D0a与两个版本S2/S4/S5共7条登记退出活动表并注明实现更正。原D1/D2两个版本及S0/S1共6行保留，新D0a和三个branchfix运行各正式登记一次，目前10行。没有添加绕过开关；旧临床模型、旧比较和D0a旧29份产物均保留。

D0a使用物理GPU3完成推理，未使用GPU0或故障GPU2；S组沿原CPU实现。已记录的控制器、训练及推理子进程均正常退出，本轮无剩余任务。

| 实验 | 当前真实运行目录 |
|---|---|
| D0a | `/home/tanyuejun/LeCen_workspace/project/runs/D0a/20261029-055704-resume-e134` |
| D1 | `/home/tanyuejun/LeCen_workspace/project/runs/D1_fixed32_60k/20261030-062653` |
| D2 | `/home/tanyuejun/LeCen_workspace/project/runs/D2_fixed32_60k/20261030-062657` |
| S0 | `/home/tanyuejun/LeCen_workspace/project/runs/S0/20261030-033551` |
| S1 | `/home/tanyuejun/LeCen_workspace/project/runs/S1/20261030-033919` |
| S2 | `/home/tanyuejun/LeCen_workspace/project/runs/S2_branchfix_20261031/20261031-085733` |
| S4 | `/home/tanyuejun/LeCen_workspace/project/runs/S4_branchfix_20261031/20261031-085733` |
| S5 | `/home/tanyuejun/LeCen_workspace/project/runs/S5_branchfix_20261031/20261031-085733` |

## 本地证据

管理根：`/data/tanyuejun/_management/LeCen/audit_corrections_20261031`。

- `protocol.json`：原设计、同种子、更正范围及冻结哈希。
- `archive/registration_retirement.json`、`archive/test_log.before.csv`：旧登记及保留行；`archive/D0a_before_correction/`保存旧bundle输出。
- `d0a_verification.json`及两份`monai_<split>_generation_receipt.json`：89/176覆盖、权重及输出身份、官方/自实现一致性。
- `clinical_verification.json`、`S2_reproduction.json`、`clinical_comparisons.json`、`detection_comparisons.json`：分支、严格复现及新配对比较。
- 新比较文件使用`runs/compare/branchfix_20261031_*`与`audit_corrected_20261031_*`，旧文件不覆盖。
- `status.json`为本轮总完成状态；`runs/index.json.audit_correction`与`parallel_registration.json.audited_implementation_correction`同步登记新结果。
