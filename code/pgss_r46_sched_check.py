"""R46：调度不影响结果的核对（T2 运行中途 Windows 把后台 worker 挤到能效核，见报告 4.9）。
把在「被挤到能效核」时段跑完的重复，在性能核上重跑到独立目录，与正式产物逐位比较。
用法：python pgss_r46_sched_check.py run <cond> <rep>   → 重跑到 t2/sched_check/runs
      python pgss_r46_sched_check.py cmp <cond> <rep> ... → 逐位比较并写 R46_SCHED_CHECK.json（含变异对照）"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
import numpy as np, pandas as pd
import pgss_r46_t2 as T2

OUT = T2.T2DIR / "sched_check" / "runs"
TIMING = {"sec_total", "sec_main", "sec_cert", "peak_rss_mb"}


def same_df(a, b):
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fc":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not (pd.Series(x).astype(str).values == pd.Series(y).astype(str).values).all():
            return False
    return True


def cmp_one(tag, mutate=False):
    o, r = T2.RUNS / tag, OUT / tag
    res = {}
    for fn in ("arms.parquet", "pairs.parquet", "sel.parquet"):
        a, b = pd.read_parquet(o / fn), pd.read_parquet(r / fn)
        if mutate:
            fc = [c for c in b.columns if b[c].dtype.kind == "f"][0]
            b = b.copy(); b.loc[b.index[0], fc] = b.loc[b.index[0], fc] * (1 + 1e-12)
        res[fn] = same_df(a, b)
    ja = json.loads((o / "rep.json").read_text(encoding="utf-8"))
    jb = json.loads((r / "rep.json").read_text(encoding="utf-8"))
    diff = sorted(k for k in set(ja) | set(jb) if k not in TIMING and ja.get(k) != jb.get(k))
    res["rep.json（除计时字段）相同"] = not diff
    res["rep.json 不同字段"] = diff
    res["原计时秒"] = ja.get("sec_total"); res["重跑计时秒"] = jb.get("sec_total")
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        OUT.mkdir(parents=True, exist_ok=True)
        T2.run_rep(sys.argv[2], int(sys.argv[3]), write=True, runs=OUT)
        print("done", sys.argv[2], sys.argv[3])
    else:
        tags = [f"{c}__r{int(r):03d}" for c, r in zip(sys.argv[2::2], sys.argv[3::2])]
        R = {t: cmp_one(t) for t in tags}
        M = {t: cmp_one(t, mutate=True) for t in tags}
        ok = all(all(v[f] for f in ("arms.parquet", "pairs.parquet", "sel.parquet", "rep.json（除计时字段）相同")) for v in R.values())
        mut_caught = all(not v["arms.parquet"] for v in M.values())
        out = dict(说明="在能效核时段跑完的重复，在性能核上重跑后与正式产物逐位比较；变异对照：重跑产物 arms 首个浮点值乘 1+1e-12 必须被判为不同",
                   逐位相同=ok, 变异对照全部检出=mut_caught, 结果=R, 变异对照=M)
        p = T2.T2DIR / "R46_SCHED_CHECK.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
