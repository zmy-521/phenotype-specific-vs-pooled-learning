# -*- coding: utf-8 -*-
r"""R47 合成示例：不需要任何受限数据，演示整套比较程序（冻结引擎开发 P* 与 HTFS → 12 个选择器 + POOLMIX → 已知真值评价）。

两种用法（都只用合成数据；输出写到 --out 指定的目录，不碰任何冻结目录）：

  python pgss_r47_example.py reproduce [--cond P2] [--rep 0] [--out DIR] [--records DIR]
      用冻结代码重新运行 T2 的一个正式重复（同一种子），与交付的逐重复记录逐位比较
      （arms / pairs / sel 三张表，及 rep 记录中除计时外的全部字段）。**不产生新的研究数字**：结果要么与已交付的记录逐位相同，要么报告差异。
      单进程约 5–8 分钟（i7-12700F）。逐位相同依赖相同的 xgboost / numpy 版本与平台；不同平台上可能只在末位浮点上不同，脚本会报告最大绝对差。

  python pgss_r47_example.py quick [--cond P2] [--rep 0] [--out DIR]
      缩小网格（PGSS_T1_SMOKE=1：每族 1 组超参、k ∈ {5, 23}、最多 60 轮）、评价固定 10 万样本的演示运行，约 2 分钟。
      **只用于检查安装与看懂输出格式，数字不可与论文比较。**

--records 默认依次查找：<本脚本上一级>/t2_records（交付 zip 的布局）、29_R47_写作交接/t2_records（项目布局）。
输出：<out>/<cond>__r<rep>/{arms,pairs,sel,fitlog}.parquet、rep.json，以及 <out>/EXAMPLE_RESULT.json（比较结果或演示摘要）。
"""
import sys, os, json, pathlib, argparse, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIMING = {"sec_total", "sec_main", "sec_cert", "peak_rss_mb"}
TABLES = ("arms", "pairs", "sel")


def no_throttle():
    """Windows 11：对本进程关闭 EcoQoS 执行速度节流（只改调度提示，不改计算；R46 §4.9 的教训）。其他平台不做任何事。"""
    if os.name != "nt":
        return False
    try:
        import ctypes, ctypes.wintypes as W

        class PPT(ctypes.Structure):
            _fields_ = [("Version", W.ULONG), ("ControlMask", W.ULONG), ("StateMask", W.ULONG)]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = W.HANDLE
        s = PPT(1, 0x1, 0)
        return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))
    except Exception:
        return False


def find_records(arg):
    for p in ([pathlib.Path(arg)] if arg else []) + [HERE.parent / "t2_records", HERE.parent / "29_R47_写作交接" / "t2_records"]:
        if (p / "rep.jsonl").exists():
            return p
    raise FileNotFoundError("找不到 t2_records（需要 rep.jsonl 与 arms/pairs/sel 的 .parquet 或 .csv.gz）；用 --records 指定")


def load_frozen(rec, tag):
    import pandas as pd
    out = {}
    for t in TABLES:
        pq, cv = rec / f"{t}.parquet", rec / f"{t}.csv.gz"
        try:
            d = pd.read_parquet(pq) if pq.exists() else None
        except Exception:                      # 无 parquet 引擎时退回 CSV（以 %.17g 写出，round_trip 读入，浮点逐位保真）
            d = None
        if d is None:
            d = pd.read_csv(cv, float_precision="round_trip")
        out[t] = d[d["重复"] == tag].drop(columns="重复").reset_index(drop=True)
    reps = [json.loads(l) for l in (rec / "rep.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out["rep"] = next(r for r in reps if r.get("tag") == tag)
    return out


def compare_df(a, b):
    """逐列比较；浮点列要求逐位相等（NaN 视为相等），并报告最大绝对差。"""
    import numpy as np, pandas as pd
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return dict(相同=False, 原因=f"形状或列不同：{a.shape} vs {b.shape}")
    worst, bad = 0.0, []
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            x, y = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float), pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
            if not np.array_equal(x, y, equal_nan=True):
                bad.append(c); worst = max(worst, float(np.nanmax(np.abs(x - y))))
        else:
            na = lambda v: "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            if not (pd.Series(x).map(na).values == pd.Series(y).map(na).values).all():
                bad.append(c)
    return dict(相同=not bad, 不同列=bad, 浮点最大绝对差=worst)


def norm(v):
    return json.loads(json.dumps(v, ensure_ascii=False, default=float))


def reproduce(a):
    import pandas as pd
    import pgss_r46_t2 as T2
    tag = T2.tag_of(a.cond, a.rep)
    rec = find_records(a.records); fz = load_frozen(rec, tag)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T2.run_rep(a.cond, a.rep, write=True, runs=out)
    new = {t: pd.read_parquet(out / tag / f"{t}.parquet") for t in TABLES}
    jn = json.loads((out / tag / "rep.json").read_text(encoding="utf-8"))
    res = {t: compare_df(fz[t], new[t]) for t in TABLES}
    jf = norm(fz["rep"])
    diff = sorted(k for k in set(jf) | set(jn) if k not in TIMING and norm(jf.get(k)) != norm(jn.get(k)))
    res["rep记录（除计时）"] = dict(相同=not diff, 不同字段=diff)
    ok = all(v["相同"] for v in res.values())
    R = dict(模式="reproduce", 重复=tag, 记录来源=str(rec), 逐位相同=ok, 结果=res, 本次秒=round(time.time() - t0, 1),
             原计时秒=fz["rep"].get("sec_total"), 说明="重新运行冻结 T2 的一个正式重复并与交付记录逐位比较；不产生新的研究数字")
    (out / "EXAMPLE_RESULT.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    return ok


def quick(a):
    os.environ["PGSS_T1_SMOKE"] = "1"
    import pandas as pd
    import pgss_r46_t2 as T2
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m, A, PR, S = T2.run_rep(a.cond, a.rep, write=True, runs=out, force_stages=0)
    s = S[S.选择器.isin(["ALWAYS_POOLED", "ALWAYS_SPLIT", "CV_POINT_M|P*", "FCT|P*", "MIX", "POOLMIX"])]
    R = dict(模式="quick（缩小网格演示，数字不可与论文比较）", 重复=m["tag"], 开发集n=m["n_dev"], 评价样本=m["n_eval"], P星=m["pstar_D"],
             本次秒=round(time.time() - t0, 1),
             候选对真值相对收益=PR[["候选对", "轨", "G", "se"]].to_dict("records"),
             选择器部署=s[["选择器", "轨", "部署", "t", "相对PstarD改善"]].to_dict("records"))
    (out / "EXAMPLE_RESULT.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 160)
    print(f"[quick] {m['tag']}：P* = {m['pstar_D']}，评价 {m['n_eval']} 样本，{R['本次秒']} 秒（缩小网格，数字不可与论文比较）")
    print(PR[["候选对", "轨", "G", "se"]].to_string(index=False))
    print(s[["选择器", "轨", "部署", "t", "相对PstarD改善"]].to_string(index=False))
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PGSS 合成示例（无需受限数据）")
    ap.add_argument("mode", choices=["reproduce", "quick"])
    ap.add_argument("--cond", default="P2")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", default=str(pathlib.Path.cwd() / "pgss_example_out"))
    ap.add_argument("--records", default=None)
    a = ap.parse_args()
    no_throttle()
    sys.exit(0 if (reproduce(a) if a.mode == "reproduce" else quick(a)) else 1)
