# LUNA16 官方评测脚本

本机已下载并验证的 Python 3 兼容入口为：

`/home/tanyuejun/LeCen_runtime/luna16_official_py3/noduleCADEvaluationLUNA16.py`

该路径已写入 `configs/data.yaml` 的 `paths.luna16_evaluation_script`，`eval/froc.py` 从配置读取。
旧运行配置没有此字段时使用项目当前数据配置的路径。
原始文件保留在 `/data/tanyuejun/LUNA16/raw/evaluation/evaluationScript/`，
兼容修改记录在运行目录的 `compatibility.json`，未改命中规则与 FROC 计算。

如果迁移到其他机器，也可以把代码放在本目录，并相应更新配置：

```
eval/luna16_official/
  noduleCADEvaluationLUNA16.py
  NoduleFinding.py
  tools/csvTools.py
```

必要时只做 Python 3 兼容修改（`print` 语句、整数除法等），不改动命中规则与 FROC 计算。

`eval/froc.py` 以子进程方式调用：

```
python noduleCADEvaluationLUNA16.py <annotations.csv> <annotations_excluded.csv> <seriesuids.csv> <preds.csv> <out_dir>/
```

并读取脚本写出的 `froc_<preds 文件名>.txt`（每行 `fps,sens,threshold`）计算 CPM。
