# D0a / D1 / D2：最终结果与完整训练日志

发布日：2026-10-30（北京时间）。本目录共11个可读文本文件，保存已完成运行的指标、置信区间、配置摘要和全量数值日志。

## 最终测试结果

| 实验 | CPM（95% CI） | 灵敏度 @1 FP/scan | 3–6 mm 灵敏度 @1 FP/scan | 评测权重 |
|---|---:|---:|---:|---|
| [D0a](D0a/results.json) | 93.79%（90.30–96.83%） | 95.07% | 92.22% | epoch 165 |
| [D1](D1/results.json) | 78.99%（73.05–84.66%） | 81.61% | 64.44% | step 28000 |
| [D2](D2/results.json) | 83.12%（77.42–87.82%） | 84.75% | 74.44% | step 20000 |

上述是**测试集结果**，均使用验证集选出的权重。`results.json` 保存完整官方/自实现指标、各尺寸组以及2000次患者级bootstrap的95% CI。CPM与官方评测一致；尺寸分组及CI由自实现计算。

本轮D2相对D1，测试CPM高4.13个百分点，3–6mm灵敏度高10.00个百分点。配对bootstrap（D1−D2）的95% CI：CPM为[−7.62, −0.65]个百分点，小结节灵敏度为[−19.10, −1.83]个百分点；详见[comparisons.json](comparisons.json)。这些结果限于本次固定划分、训练预算与种子。D0a的架构、批量及训练协议不同，不能将差异归因于单一模块。

## 完整日志与训练预算

| 实验 | 训练记录 | 验证记录 | 已完成预算 | 实际optimizer updates | 全局batch / 卡数 |
|---|---|---|---|---:|---|
| D0a | [train_log.csv](D0a/train_log.csv) | [val_metrics.csv](D0a/val_metrics.csv) | 200 epoch；31,200 iterations | 31,186 | 16 / 4 |
| D1 | [train_log.csv](D1/train_log.csv) | [val_metrics.csv](D1/val_metrics.csv) | 30,000 updates | 30,000 | 4 / 2 |
| D2 | [train_log.csv](D2/train_log.csv) | [val_metrics.csv](D2/val_metrics.csv) | 30,000 updates | 30,000 | 4 / 2 |

CSV保留全部逐步记录，可用GitHub的Raw入口或下载后阅读。D1/D2每2000步记录一次验证；`step`即成功optimizer更新次数，`grad_norm`是裁剪前的总梯度范数。D0a的`iteration`包含14次AMP跳过，`optimizer_step_applied`为实际更新标记。

D0a以最后完整运行`20261029-055704-resume-e134`为准。训练CSV由历史epoch1–134（iteration1–20904）与续训epoch135–200（iteration20905–31200）按顺序合并，只有一个表头，无重复或缺失。`elapsed_seconds`在iteration20905恢复时重置，保留原值，不是连续墙钟。验证CSV合并为epoch5、10……200的40条记录。恢复包含优化器/调度器/scaler状态，但未保存随机数状态，不能声称逐步完全一致。

D0a训练监控`val_coco`为MONAI返回各COCO指标的均值，并非CPM或单项mAP；最终CPM见`results.json`。D1/D2共享初始化、固定划分、种子、卡数及30k更新预算，仅恢复分支开关不同。D1/D2的最佳权重分别来自28k/20k，这不改变二者已完成30k训练的事实。

## 验证复核与来源

D1/D2训练完成后复核了最佳及三万步权重，修复float32概率与float64阈值比较的同分边界；最低阈值严格大于，LUNA FROC工作点包含阈值等分候选。修复没有改变训练参数。四次真实验证的指标和权重排序未改变，选定权重的官方CPM与自实现一致。原始训练验证CSV保留当时数值；修复后的验证工作点信息位于各自`results.json`的`posttraining_validation`中。累计recall chain包含前置热图命中条件，不能直接当作最终FROC灵敏度。

D0a运行快照记录的仓库基准为`e497537194cddf7bfec375c1064e524f8a9bee16`；D1/D2的基准为`3522fd9939eed4a12daac049bfcd948e72cdc068`，并包含当时工作区的BF16/数值修复。每组`results.json`记录运行ID、配置、种子、原始文件SHA256及当时源快照指纹；本提交发布的是结果与日志。完整模型权重、预测CSV、bootstrap明细及运行源码快照保存在本机原始`runs/<实验>/<运行ID>/`。

测试只用于最终报告，不根据测试分数调参或重训。Bootstrap CI仅体现患者抽样不确定性，不包含重复训练、换种子或换划分的波动。
