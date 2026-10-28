# D1/D2 检测训练数值错误报告（2026-10-28）

## 状态与影响

截至 2026-10-28 20:22（北京时间，UTC+8），本轮 D1、D2 均已异常退出，训练未完成。
错误为 `ValueError: matrix contains invalid numeric entries`，来自匈牙利匹配的
`scipy.optimize.linear_sum_assignment`。历史加速分支增加过诊断能力，**尚未修复数值错误**。
本次在 `main` 仅归档本报告与机器可读证据；加速代码、诊断代码和新增测试均不并入 `main`。
此前的正常运行耗时估算不包含故障排查和重跑时间，不能继续作为完成时间承诺。

| 实验 | 运行 ID（服务器以 UTC 命名） | GPU | 最后成功更新 | 目标更新数 | launcher 报错时间（北京时间） | 最后验证／普通检查点 |
|---|---|---|---:|---:|---|---:|
| D1 | `20261028-105142` | 0、1 | 3,137 | 60,000 | 2026-10-28 19:05:31 | 2,000 |
| D2 | `20261028-110203` | 4、5 | 3,694 | 60,000 | 2026-10-28 19:38:14 | 2,000 |

失败发生在表中最后成功更新之后的下一次前向／损失计算；失败 batch 没有产生新的 optimizer step。
D1 的累计 AMP 跳过次数为 10，D2 为 9。普通 `best.pt` 与 `last.pt` 的内部 `step` 均已核验为 2,000。
日志中没有本次失败由 CUDA OOM 或显存限制触发的证据。

## 历史加速分支的改动范围

基线为 `17ac9e6efd95b00fb32500dfd847058ea0488f38`，当时工作分支为 `perf/inference-batch`。
运行时改动尚未提交，因此历史运行的 `hashes.json` 仍指向该基线；各运行另保存了源码清单和补丁。
这些改动后来上传为提交 `f9a1c41fdaa23f9102f17ee13ff971bd7de99dcc`。
以下为该历史实验的改动记录，不是 `main` 中已启用的功能：

- `engine/infer_det.py` / `configs/det/base.yaml`：增加独立的 `infer.batch`；默认 `null` 时保留原来的 `train.batch` 行为，支持尾批次并检查非法值。
- `engine/train_det.py`：可选 `train.failure_dir`，在前向／损失计算抛出 `ValueError` 或 `RuntimeError` 时保存现场后重新抛出异常。
- `tests/test_infer_batch.py` / `tests/test_failure_snapshot.py`：覆盖批次拆分、坐标及 NMS 结果、参数校验，以及现场保存和默认关闭行为。
- 历史分支 README：补充 step／epoch、推理 batch 和检查点限制说明。

模型结构、损失公式、学习率、优化器、全局训练 batch、采样策略和停止条件保持原样。
服务器显卡调度、显存监控和 NCCL 启动环境位于仓库外的 runtime；本报告记录其实际配置。
原工作目录 `LeCen` 的 79 个受版本控制文件在发布前再次核验，无修改。

## epoch 与 step 的实际含义

检测训练在 `engine/train_det.py` 中执行 `while step < cfg.train.iters`。
仅在 `optimizer_step` 确认发生有效更新后递增 `step` 和调度器；AMP 溢出跳过更新时不计步。
验证与保存普通检查点每 2,000 次有效更新执行一次。

`data/det_dataset.py:PatchSampler` 无限、有放回地抽取病例和 patch，按全局样本下标分配给 DDP rank。
其内部 epoch 每生成 `n_cases` 个全局样本递增，只用于采样周期与增强种子，并非完整遍历所有 CT。
本次 623 个训练 CT、全局 batch 4、`accum_steps=1`：不计 AMP 跳过，
60,000 次更新消耗 240,000 个 patch，等于约 385.2 个采样周期。
跳过的 batch 和 DataLoader 预取会增加已消费或预生成的样本数。

`engine/train_surv.py` 的临床生存训练则按 epoch 执行，默认最多 200 个 epoch、早停 patience 20；本轮尚未启动。

## 环境与运行参数

| 项目 | 实际配置 |
|---|---|
| Python / PyTorch | Python 3.10；PyTorch 2.7.1+cu128 |
| GPU | NVIDIA RTX PRO 6000 Blackwell，96GB；每个实验 2 个 DDP rank |
| NumPy / SciPy / pandas | 1.23.5 / 1.10.1 / 2.0.3 |
| MONAI / SimpleITK / OmegaConf | 1.3.2 / 2.5.6 / 2.3.0 |
| NCCL | 启动时局部预加载 2.31.2，`NCCL_P2P_DISABLE=1`；`pip freeze` 中的基础包为 2.26.2，不能据此认定实际加载版本 |
| 数据划分 | LUNA16：623 train、89 val、176 test；未使用 test 调参 |
| Patch / batch | `(80, 192, 192)`；全局 batch 4，每 rank 2，累积 1；workers 8/rank |
| 优化器 | AdamW；backbone LR 1e-4、decoder LR 2e-4、weight decay 1e-4、clip norm 1 |
| 调度 / 精度 | warmup 1,000 次更新后 cosine；FP16 autocast + GradScaler |
| 种子 | 共享初始化 1000、D1 recovery 初始化 1001、采样器 2000 |
| 验证 | 每 2,000 次更新；两项失败运行均使用 `infer.batch=4` |
| D1 显存配置 | GPU 0/1 启动前确认空闲；PyTorch 分配器上限 42GiB，进程显存监控阈值 48,000,000,000 字节 |
| D2 显存配置 | GPU 4/5 使用可用资源；分配器 46GiB，进程监控阈值 49GiB |

现有服务器运行命令的结构如下。重跑需先核对数据路径、可用显卡与上述 runtime 环境；这些命令不会自动恢复优化器状态。

```bash
torchrun --standalone --nproc_per_node=2 -m engine.train_det \
  --cfg /home/tanyuejun/LeCen_runtime/configs/parallel/D1.yaml
torchrun --standalone --nproc_per_node=2 -m engine.train_det \
  --cfg /home/tanyuejun/LeCen_runtime/configs/parallel/D2.yaml
```

## 错误调用链与已知边界

两次失败均沿以下调用链退出（省略 DDP 与 Python 启动器帧）：

```text
engine/train_det.py: train -> TrainStep.forward -> compute_loss
losses/matching.py: match_layers -> hungarian_match
    row, col = linear_sum_assignment(cost.detach().float().cpu())
ValueError: matrix contains invalid numeric entries
```

可确认的是交给 SciPy 的代价矩阵存在不被接受的数值；尚未确认最先出现异常的是模型输出、框变换、
分类代价、L1 代价还是 GIoU 代价，不能据此直接归因于 AMP、数据损坏或某个具体算子。
现有 GradScaler 可以跳过反向梯度溢出的更新，但此次异常发生在前向匹配，尚未到达该处理阶段。

此前未修改代码的 D1 在第 3,466 步后出现相同错误；首轮并行 D2 在第 2,673 步后出现相同错误。
这说明仅移动到空闲 GPU 或重新启动没有解决问题。
先前用第 2,000 步权重检查旧 D1 失败采样得到有限损失；该检查不是用实际失败时的权重复现，不能排除问题。

D1 在 GPU 0/1 上是相同配置、种子、初始权重的重新训练。与 GPU 6/7 的旧副本首步损失一致，
后续曲线未通过 `rtol=1e-2, atol=1e-4` 检查；因此这次切换不属于精确续训。
新副本到 2,361 步、旧副本到 2,353 步时采用新副本，旧日志和检查点保留。

## 已保存证据与下一步定位

机器可读摘要见 [training-failure-evidence.json](2026-10-28-training-failure-evidence.json)。服务器本地证据：

- D1 日志：`/data/tanyuejun/_management/LeCen/logs/train_D1_idle.log`。
- D2 日志：`/data/tanyuejun/_management/LeCen/logs/train_D2_restart.log`。
- 运行记录：`/home/tanyuejun/LeCen_accel/runs/D1/20261028-105142/` 和 `runs/D2/20261028-110203/`，含配置、种子、环境、训练日志、验证指标和检查点。
- 失败现场：`/data/tanyuejun/_management/LeCen/failures/`，具体文件名、大小和 SHA-256 见摘要。

失败现场包含该 rank 的模型、recovery（D1）、optimizer、scheduler、scaler、当前 batch、
最后成功 step、累计 skipped、错误字符串、保存时的 torch/CUDA RNG 状态。
它们适合做失败 batch 的前向诊断；并未记录所有 rank 的完整采样器／预取状态，也没有实现恢复入口，
不能宣称能够精确恢复整个 DDP 训练。普通 `best.pt` / `last.pt` 只含模型、recovery 和 step。

下一步应使用现场中的实际模型与 batch，检查输入、GT、每层预测及三项匹配代价的有限性，
定位首个无效值，并对比 FP16 与 FP32 的同批次前向。应建立能复现该问题的回归检查后再修复和恢复训练。
本次提交没有把异常值替换为常数，也没有跳过出错样本来掩盖失败。

本次 `main` 上传内容仅包含本报告和汇总证据；影像、病例标识、batch 张量、权重及大型现场文件保留在服务器。

## 验证

历史加速分支发布前执行以下 CPU 回归检查（GPU 不参与）：**21 项通过，耗时 2.45 秒**；
`git diff --check` 通过。结果另记于机器可读摘要的 `publication_validation` 字段。
其中新增的两份测试文件未并入 `main`，以下命令应在历史实验代码上执行：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider \
  tests/test_infer_batch.py tests/test_failure_snapshot.py tests/test_amp_step.py \
  tests/test_matching.py tests/test_empty_cases.py \
  tests/test_determinism.py::test_sampler_sequence \
  tests/test_determinism.py::test_ddp_partition_matches_single
git diff --check
```

这些测试验证批处理接口、诊断保存、AMP 更新计数、基础匹配与采样行为，不代表 60,000 步训练已通过。
完整训练的同种子曲线一致性测试未通过本次长跑验证，不作可复现性保证。

另有一次固定第 2,000 步权重、89 例验证 CT 的历史推理测速：batch 4→8 时，
212.88s→192.19s，吞吐约 1.108 倍；三项汇总 FROC 指标相同，但 5 例候选数组有差异，
不能认定逐元素等价。因此本轮两项训练均保留推理 batch 4。
