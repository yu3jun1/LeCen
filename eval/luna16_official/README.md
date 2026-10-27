# LUNA16 官方评测脚本

把 LUNA16 `evaluationScript.zip` 中的评测代码原样放在本目录：

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
