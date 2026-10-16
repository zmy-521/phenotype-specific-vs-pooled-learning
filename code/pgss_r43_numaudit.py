# -*- coding: utf-8 -*-
r"""报告数字溯源检查（R43 版）及其检出力。

R42 的做法：md 中 d 位小数的数字 s，只要证据表**任何**单元格的值 v 满足 round(|v|,d)==s 或 round(100|v|,d)==s 即算找到。
问题：证据表含逐单元、全对照等明细表时数值池很大，2–4 位小数的随机数几乎都能「找到」——检查对低位数数字没有检出力。
R43 的做法：
  1. 数值池只用**报告引用的汇总表**（由报告脚本在「目录」表的「报告引用」列声明）；
  2. 同时报告检出力：在 [0, 2) 内均匀抽 5000 个 d 位小数，计算被误判为「找到」的比例（随机命中率），按 d = 1…6 给出；
  3. 科学计数法数字按同一格式（同位数尾数）匹配。
用法：python pgss_r43_numaudit.py r42   复查 R42 报告（汇总表 = 除明细表外的全部表），写 readonly/R42_NUMAUDIT.json
"""
import sys, re, json, pathlib
import numpy as np, pandas as pd

SCRUB = (r"(?im)(xgboost\s*\d+(\.\d+)+|\bv\d+\.\d+|xgb-\d+\.\d+|`[^`]*`|https?://\S+|\b\d+\.\d+\.\d+\b|\bR\d+\.\d+\b"
         r"|^#{1,6}\s+[0-9A-Z.]+|§[0-9A-Z.]+)")  # 版本、代码、网址、轮次标签、标题编号、章节引用


def pool(xl, names):
    a = []
    for k in names:
        num = xl[k].apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel()
        a.append(num[np.isfinite(num)])
    return np.unique(np.abs(np.concatenate(a))) if a else np.array([])


def audit(md_text, xl, names, seed=0):
    md = re.sub(SCRUB, " ", md_text)
    v = pool(xl, names)
    sci = re.findall(r"(?<![\w.])(\d\.(\d+)e[-+]\d+)", md)
    miss = [s for s, dec in sci if s not in {f"{x:.{len(dec)}e}" for x in v}]
    md2 = re.sub(r"(?<![\w.])\d\.\d+e[-+]\d+", " ", md)
    nums = re.findall(r"(?<![\d.])(\d+\.(\d{1,8}))(?![\d.])", md2)
    sets = {d: set(np.round(v, d)) | set(np.round(v * 100, d)) for d in range(1, 9)}
    miss += sorted({s for s, dec in nums if round(float(s), len(dec)) not in sets[len(dec)]})
    rng = np.random.default_rng(seed)
    power = {d: float(np.mean([x in sets[d] for x in np.round(rng.uniform(0, 2, 5000), d)])) for d in range(1, 7)}
    bydec = pd.Series([len(d) for _, d in nums]).value_counts().sort_index().to_dict()
    return dict(数值池大小=int(len(v)), 带小数数字=len(nums), 科学计数法数字=len(sci), 位数分布={int(k): int(c) for k, c in bydec.items()},
                找不到=miss, 随机命中率={int(k): round(c, 4) for k, c in power.items()})


def r42():
    root = pathlib.Path(__file__).resolve().parent.parent
    S = root / "23_R42_正式主实验"
    md = (S / "R42_正式主实验_统一报告.md").read_text(encoding="utf-8")
    xl = pd.read_excel(S / "R42_正式主实验_证据表.xlsx", sheet_name=None)
    detail = ["B3_逐单元", "C2_全部对照", "D3_认证逐单元", "E1_学习曲线", "A6_不可执行定位", "A7_不稳定单元"]
    out = dict(R42原检查_全部表=audit(md, xl, list(xl)), 本轮复查_只用汇总表=audit(md, xl, [k for k in xl if k not in detail]),
               明细表=detail)
    p = root / "24_R43_结果补齐与T1" / "readonly" / "R42_NUMAUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["r42"]:
        r42()
