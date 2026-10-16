# -*- coding: utf-8 -*-
r"""R43 闭环核对的变异对照：每一项核对都必须能在被故意破坏的产物上失败。

做法：把一个任务（mimic × 住院死亡 × 女性）的 75 个单元目录复制到临时目录，逐项施加一种破坏，
用同一个 `task_pass` 重新核对，要求目标核对项出现 False；未破坏的副本必须全部为 True（基线）。
为了让破坏只命中目标项，除「产物指纹」一项外，破坏后都按新文件重写运行记录里的指纹。

用法：python pgss_r43_readonly_tests.py <临时目录>
输出：24_R43_结果补齐与T1/readonly/R43_READONLY_TESTS.json
"""
import sys, json, pathlib, shutil, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r43_readonly as RO
from pgss_xgb_run import fsha

TASK = ("mimic", "OUT_HOSP_MORT", "女性")
SEED, FOLD = RO.SEEDS[0], 2


SRC = RO.RUNS                                   # 原始产物（只读）


def fresh(tmp):
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pre = "__".join(TASK) + "__"
    for d in SRC.iterdir():
        if d.name.startswith(pre):
            shutil.copytree(d, tmp / d.name)


def reseal(od):
    m = json.loads((od / "meta.json").read_text(encoding="utf-8"))
    m["artifacts"] = {f.name: fsha(f) for f in sorted(od.glob("*.parquet"))}
    (od / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def unit(tmp, branch, frac=None):
    return tmp / RO.R42I.tag(*TASK, SEED, FOLD, branch, frac)


def edit_preds(od, fn, name="preds_test.parquet", seal=True):
    P = pd.read_parquet(od / name); P = fn(P); P.to_parquet(od / name, index=False)
    if seal:
        reseal(od)


def flip_y(P):
    P.loc[P.index[0], "y"] = 1 - P.loc[P.index[0], "y"]; return P


def set_one(P):
    c = [x for x in P.columns if x not in ("unit_id", "y", "z")][0]; P.loc[P.index[0], c] = 1.0; return P


def bump_nsub(od):
    m = json.loads((od / "meta.json").read_text(encoding="utf-8")); m["n_sub"] += 1
    (od / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


MUTATIONS = [
    ("main 删一行", "测试集合一致", lambda t: edit_preds(unit(t, "main"), lambda P: P.iloc[1:])),
    ("main 重复一行", "单位唯一", lambda t: edit_preds(unit(t, "main"), lambda P: pd.concat([P.iloc[:-1], P.iloc[:1]]))),
    ("main 翻转一个 y", "y一致", lambda t: edit_preds(unit(t, "main"), flip_y)),
    ("main 预测改为 1.0", "预测在合法范围", lambda t: edit_preds(unit(t, "main"), set_one)),
    ("main 改预测不更新指纹", "产物指纹一致", lambda t: edit_preds(unit(t, "main"), set_one, seal=False)),
    ("cert 认证集删一行", "认证集合一致", lambda t: edit_preds(unit(t, "cert"), lambda P: P.iloc[1:], name="preds_C.parquet")),
    ("lc 子集规模记录 +1", "子集规模与记录一致", lambda t: bump_nsub(unit(t, "lc", 0.5))),
]


def run_check(tmp):
    RO.RUNS = tmp
    try:
        _, rows, *_ = RO.task_pass(*TASK)
        return pd.DataFrame(rows), None
    except Exception as e:                      # 破坏导致核对流程本身拒绝继续，也算检出
        return None, f"{type(e).__name__}: {e}"


def main(tmp):
    tmp = pathlib.Path(tmp); orig = RO.RUNS; t0 = time.time(); res = []
    fresh(tmp)
    U, err = run_check(tmp)
    bool_cols = [c for c in U.columns if set(U[c].dropna().unique()) <= {True, False} and c not in ("失败记录",)]
    base_ok = bool(err is None and all((U[c].dropna() == True).all() for c in bool_cols))  # noqa: E712
    res.append(dict(变异="基线（未破坏）", 目标项="全部", 检出=None, 基线全部通过=base_ok))
    for name, col, fn in MUTATIONS:
        fresh(tmp); fn(tmp)
        U, err = run_check(tmp)
        if U is None:
            res.append(dict(变异=name, 目标项=col, 检出=True, 方式=f"流程拒绝：{err[:120]}")); continue
        hit = U[col].eq(False).any() if col in U else False
        others = [c for c in bool_cols if c != col and c in U and U[c].eq(False).any()]
        res.append(dict(变异=name, 目标项=col, 检出=bool(hit), 其他同时失败项=",".join(others)))
    RO.RUNS = orig
    shutil.rmtree(tmp, ignore_errors=True)
    out = dict(全部检出=all(r["检出"] for r in res[1:]), 基线全部通过=base_ok, 明细=res, 秒=round(time.time() - t0))
    (RO.OUT / "R43_READONLY_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
