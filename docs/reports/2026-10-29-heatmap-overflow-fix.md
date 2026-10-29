# D1 / D2 热图卷积 FP16 溢出修复

## 根因与现场证据

用失败现场中的实际模型与原始 batch，在 GPU 上重新执行前向，两组均稳定复现
`ValueError: matrix contains invalid numeric entries`。输入影像、tokens、坐标、GT 和模型参数均为有限值。

首次非有限输出位于 `heatmap.conv.0`（256 通道的 3×3×3 Conv3d），发生在注意力之前。
对同一份卷积输入和权重禁用 autocast，输出保持有限：

| 实验 | 保存的已更新步数 | 卷积输入范围 | 同输入 FP32 卷积最大值 | 原 FP16 输出 Inf 数量 |
| --- | ---: | --- | ---: | ---: |
| D1 | 15441 | [-400.75, 397.75] | 67304.515625 | 4 |
| D2 | 10457 | [-677.5, 659.5] | 65835.0546875 | 1 |

这些值超过 FP16 最大有限值 65504。卷积先生成 Inf，紧接的 GroupNorm 将异常扩散为 NaN，
再污染热图、尺寸、decoder 输入和匹配代价。此前的注意力分数修复仍保留；
这两份新现场的直接起点是独立的卷积输出溢出。GradScaler 只调整反向损失缩放，无法修复这类前向溢出。

## 实现

`models/heatmap_head.py` 的 `forward` 在共享的 `Conv3d → GroupNorm → ReLU` 区域禁用 autocast，
并把输入转换为 FP32；归一化后的 1×1×1 分类、尺寸输出头继续使用调用者的 AMP 设置。
只改计算精度，网络层、参数名称与形状、损失、数据采样及优化器超参数不变。
没有把 NaN 替换成常量、裁剪匹配代价或跳过失败样本。

## 验证

1. 同一失败模型与 batch：修复前两组都报原错误；修复后所有层预测框和匹配代价有限。
   完整损失 D1 为 6.086928，D2 为 21.420540。
2. 恢复现场优化器、scheduler 和 scaler，在诊断副本上各完成 4 次有效更新：
   损失、有效更新梯度范数与参数均有限。该检查只用于验证，不作为正式训练续训。
3. 新回归测试使用有限输入构造卷积中心输出 108000，旧 FP16 路径出现 Inf/NaN；
   修复路径前向、输入梯度与参数梯度全部有限。常规 FP32 输出及输入梯度与原计算相同。
4. 新回归、原注意力与匹配检查共 9 项通过；D1、D2 各完成两卡、真实数据、全局 batch 4 的
   16 次有效更新，两个 rank 更新次数一致。初始 AMP loss scale 回退不计为更新。
5. 扩展测试中 33 项通过；4 项真实数据集成测试因仓库模板路径 `???` 或默认
   `runs/init/init.pt` 未配置而失败。生存标签测试因运行环境缺少 `pycox` 无法收集。
   本次真实数据回放与 DDP 验证使用外部已对齐配置，不把完整测试集报告为全通过。

短测和失败回放证明已修复这两份现场，不等于 60000 步长跑已经完成。

## 实验与材料

D1、D2 均以同一份共享初始化从头训练，保持两卡、全局 batch 4、60000 次有效更新、
每 2000 步验证，以及原学习率日程。共享初始化 SHA-1 为
`fe9f505170acd3add1916c34523b91549bb0ea1a`。保留旧实验与失败现场，D0a 继续独立运行。

现场：`/data/tanyuejun/_management/LeCen/ssd_failures/`。
证据：`/data/tanyuejun/_management/LeCen/numerical_fix_20261029/`：

- `before_D1.json`、`before_D2.json`：原代码重放；
- `trace_D1.json`、`trace_D2.json`：逐算子溢出与 FP32 卷积对照；
- `after_D1.json`、`after_D2.json`：修复后重放；
- `backward_D1.json`、`backward_D2.json`：现场反向与优化器验证；
- `ddp_D1.json`、`ddp_D2.json`：两卡真实数据短测；
- `source_before/`：修复前源码及已有工作区改动。

外部诊断脚本位于 `/home/tanyuejun/LeCen_runtime/`：
`trace_backbone_failure.py`、`replay_failure_snapshot.py`、`verify_saved_failure_updates.py`、
`validate_fixed_training.py`。正式运行入口分别为 `runs/D1/current` 和 `runs/D2/current`；
后续评估由 `/data/tanyuejun/_management/LeCen/experiment_queue/plan.json` 登记。
