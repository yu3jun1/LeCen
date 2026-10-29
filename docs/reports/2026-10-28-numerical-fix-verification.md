# 真实失败现场的数值修复验证

## 结论

对 D1 step 3137、D2 step 3694 的已保存模型与原始失败 batch，在 GPU FP16 下重放：
`17ac9e6` 都复现 `ValueError: matrix contains invalid numeric entries`；
修复版本 `9ea85d2` 两份都通过，完整损失分别为 16.792885、14.112576。

这两次实际失败的直接原因是第一层 decoder 自注意力的 FP16 分数溢出。
尺寸指数导致 GIoU 溢出是可用合成输入复现的独立风险，但不是这两份现场的起点。

## 证据

模型参数、影像、tokens、坐标和 GT 均为有限值。热图 log 尺寸与第一处异常如下：

| 检查 | D1 | D2 |
|---|---:|---:|
| 热图 size logit 最大值 | 3.574219 | 4.210938 |
| 第一处异常模块 | `layer0.self_attn` | `layer0.self_attn` |
| 同一投影 Q/K 用 FP32 计算的分数最大值 | 84,966.10 | 68,365.88 |
| 该处 FP16 分数中的 inf 数量 | 2 | 2 |

FP16 最大有限值为 65,504。第一层注意力输出首先产生 NaN，第一层框包含一条无效 query，
后续层再把 NaN 传播到同一样本的全部 query。可用的框 log 尺寸约在 1.3–4.5 之间，
未观察到约 30 的有限 log 尺寸或先于注意力出现的框体积溢出。

## 分项对照

所有测试均使用同一失败现场的实际模型与 batch，未进行权重更新。

| 代码／计算方式 | D1 | D2 |
|---|---|---|
| 旧代码，FP16 | 原错误复现 | 原错误复现 |
| 仅加入尺寸约束，强制原注意力权重路径 | 原错误复现 | 原错误复现 |
| 旧代码仅改 `need_weights=False` | 有限损失 16.792881 | 有限损失 14.112574 |
| 旧代码，FP32 | 有限损失 16.786831 | 有限损失 14.125857 |
| 尺寸约束 + `need_weights=False`，FP16 | 有限损失 16.792885 | 有限损失 14.112576 |

因此保留两项修复：尺寸约束阻止已知指数风险，SDPA 注意力路径解决这两份现场实际触发的错误。
没有使用 `nan_to_num`、替换匹配代价或跳过失败样本。
不返回注意力权重的数学目标保持一致，但计算路径和浮点舍入改变，不能承诺逐位相同。

## 标签范围与回归检查

仅审计训练集：623 个 CT、852 个实例，GT 尺寸最小值 `(1, 4, 4)`、最大值 `(33, 61, 59)` 体素，
均在 `[1, (160, 384, 384)]` 内。临床肿瘤尚未完成此项资格与范围审计；该结论不外推到临床队列。
测试集未用于调参或选择修复方案。

- 相关基础 CPU 检查：22 项通过。
- 尺寸与注意力稳定性检查：7 项通过，包含实际 CUDA FP16 回归；该 GPU 测试的旧显式权重路径产生非有限输出，修复路径输出有限。
- 现场保存和 AMP 更新计数检查：4 项通过。此集合与基础检查存在重叠，不累加为独立测试总数。
- 正常 FP32 输入的 decoder 输出在两种注意力路径间通过 `rtol=1e-5, atol=1e-6` 检查。
- 尺寸约束测试覆盖极大／极小 log 尺寸、正常区间梯度和 NaN 不被隐藏的行为。

源码保持同一套 D1/D2 训练超参数；两项实验须从同一份共享初始化重新训练，
不从旧模型检查点恢复 AdamW 状态。共享初始化 SHA-1：`fe9f505170acd3add1916c34523b91549bb0ea1a`。
此前初始 loss scale 下的机制梯度诊断可能出现 NaN，不能把该记录当作有效机制结论；
这与本报告确认的前向匹配失败是两项不同检查。

## 复现材料

本地脚本：`/home/tanyuejun/LeCen_runtime/replay_failure_snapshot.py`。
旧代码单独解包至 `/tmp/lecen_failure_baseline_17ac9e6`；当前修复位于 `experiment-code` 分支。
脚本通过外部 `run_fixed_module.py` 选择源码目录并限制显存。

逐层尺寸、体积、分类／L1／GIoU 代价和注意力数值摘要保存在
`/data/tanyuejun/_management/LeCen/numerical_fix/`：

- `old_D1.json`、`old_D2.json` 与 `fixed_D1.json`、`fixed_D2.json`：直接重放。
- `old_instrumented_D1.json`、`old_instrumented_D2.json`：注意力分数检查。
- `bounds_only_*.json`、`attention_only_*.json`、`old_fp32_*.json`：分项对照。
- `ddp_preflight_D1.json`、`ddp_preflight_D2.json`：实际两卡、真实训练数据的短测结果（生成后以其中 `passed` 字段为准）。

这些短测和现场重放不等于 60,000 步长跑已经完成。正式训练状态以
`/data/tanyuejun/_management/LeCen/pipeline_status.json` 为准。
