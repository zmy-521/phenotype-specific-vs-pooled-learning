# -*- coding: utf-8 -*-
"""R36：在 R35 基础上换用 pgss_r36_engine（校准层分组隔离 + 共同预处理 +
随机顺序分组切分），并把缓存键从「文件名存在」改为「代码+数据+配置+划分哈希」。

与 R34 的差别
  - 引擎换成 `pgss_r36_engine`（分组贯穿三层 / 折内预处理 / Hard 回退已选 P* / 两侧 S-Scf 对称 / 统一坐标）
  - 结局换成已授权口径：**OUT_HOSP_MORT（住院死亡）** 与 **OUT_LOS72（ICU>72h）**
  - 特征恢复性别与年龄（#19）
  - **逐就诊保存预测与损失**（#28）：unit_id / group_id / hospital_id / y / 表型 / 各候选预测 /
    所选共享臂 / specialist 版本 / λ* / 划分哈希
  - 外层划分哈希与两两重合率落盘（#18）

用法： python pgss_r36_bench.py --db mimic|eicu [--jobs 6]
"""
import os as _os, sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent))
import pgss_threads  # noqa: F401  线程固定，必须在 numpy 之前
import hashlib, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import pgss_core as C, pgss_r36_engine as E
import hashlib as _hl, json as _json

import pgss_dataver as _DV           # 活动数据版本的单一真相源
F = _DV.DIR                          # 目录与缓存键里的版本号**必须同源**
OUTD = HERE.parent / "19_R38" / "bench"; OUTD.mkdir(parents=True, exist_ok=True)
PRED = OUTD / "preds"; PRED.mkdir(exist_ok=True)
SEEDS = [20261003, 20261004, 20261005]
K_OUTER = 5
OUTCOMES = ["OUT_HOSP_MORT", "OUT_LOS72"]
PHENOS = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
MIN_N, MIN_EV = 500, 50


def load(db):
    X = pd.read_parquet(F / f"R38_{db}_X.parquet")
    P = pd.read_parquet(F / f"R38_{db}_PHENO.parquet")
    OK = pd.read_parquet(F / f"R38_{db}_PHENOOK.parquet")
    meta = pd.read_parquet(F / f"R38_{db}_META.parquet")
    qc = pd.read_excel(HERE.parent / "16_R35_FIX" / "R35_DATA_QC.xlsx",
                       sheet_name="4_最终特征池")
    feats = [v for v in qc.特征 if v in X.columns]
    Xf = X[feats].copy()
    Xf["性别F"] = (meta.sex == "F").where(meta.sex.isin(["F", "M"]))   # 未知 → NaN（#20）
    return Xf, P, OK, meta



def joint_groups(patient, hospital):
    """患者-医院**连通分量**作为分组键（GPT 第35轮 §5.2）。

    eICU 外层要"未见医院"，但 2,071 名患者（2.18%）跨多家医院；
    只按 hospital 分折会让患者跨折，只按 patient 分折又丢掉跨医院外推。
    对二分图 (患者 — 医院) 求连通分量，按分量分折 ⇒ **两个约束同时满足**。
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for a, b in zip(patient, hospital):
        union(("P", str(a)), ("H", str(b)))
    return np.array([str(find(("P", str(a)))) for a in patient])


def cache_key(db, oc, ph, seed):
    """正式缓存键（GPT 第 38 轮 §A2）：至少绑定 11 项。

    R36 的键只哈希 pgss_core/pgss_r36_engine 与 X/PHENO/META，
    遗漏 pgss_sharing、PHENOOK、决定特征池的 QC 文件、划分算法版本。
    """
    import pgss_sharing as _SH
    fp = C.run_fingerprint((C, E, _SH))
    code = "|".join(v for k, v in sorted(fp.items()) if k.endswith("__sha256"))
    dh = []
    for p in (F / f"R38_{db}_X.parquet", F / f"R38_{db}_PHENO.parquet",
              F / f"R38_{db}_PHENOOK.parquet", F / f"R38_{db}_META.parquet",
              HERE.parent / "16_R35_FIX" / "R35_DATA_QC.xlsx",
              F / "DATA_VERSION.json"):
        dh.append(_hl.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "NA")
    cfg = _json.dumps(dict(outcomes=OUTCOMES, phenos=PHENOS, k=K_OUTER,
                           min_n=MIN_N, min_ev=MIN_EV,
                           split_algo="random-order-least-loaded-bin",
                           split_version="r36.1", data_version=_DV.ACTIVE),
                      ensure_ascii=False, sort_keys=True)
    raw = f"{code}|{'-'.join(dh)}|{cfg}|{db}|{oc}|{ph}|{seed}"
    return _hl.sha256(raw.encode()).hexdigest()[:16]


def _one_impl(db, oc, ph, seed):
    key = cache_key(db, oc, ph, seed)
    tag = f"{db}__{oc}__{ph}__{seed}__{key}"
    f = OUTD / f"{tag}.parquet"
    if f.exists(): return "SKIP_DONE"
    X, P, OK, meta = load(db)
    e = meta.eligible.fillna(False).to_numpy(bool)
    y = meta[oc].fillna(False).to_numpy(bool).astype(int)
    m = e & OK[ph].to_numpy(bool)
    idx = np.flatnonzero(m)
    if len(idx) < MIN_N or y[idx].sum() < MIN_EV: return "SKIP_TOO_SMALL"
    XX = X.to_numpy(float)[idx]              # **保留 NaN**，由引擎在折内插补（#16）
    yy = y[idx]; zz = P[ph].to_numpy(bool)[idx]
    pid = meta.group_id.astype(str).to_numpy()[idx]
    if db == "mimic":
        gg = pid                                   # MIMIC 单中心：分组键 = 患者
    else:
        hid = meta.hospital_id.astype(str).to_numpy()[idx]
        gg = joint_groups(pid, hid)                # eICU：患者-医院连通分量
    folds = E.group_split(XX, yy, zz, gg, K_OUTER, seed)
    outer_hash, canon = E.split_fingerprint(gg, folds)
    lab = np.full(len(idx), -1)
    for k, (_, te) in enumerate(folds): lab[te] = k
    preds = {p: np.full(len(idx), np.nan) for p in E.PROGS}
    infos = []
    _diag_rows = []; _diag_sums = []
    for kf, (dev, te) in enumerate(folds):
        if set(np.unique(gg[dev])) & set(np.unique(gg[te])):
            raise E.FoldFailure("外层分组泄漏")
        if set(np.unique(pid[dev])) & set(np.unique(pid[te])):
            raise E.FoldFailure("外层患者泄漏")       # eICU 同一患者跨医院也要挡（#17）
        E.diag_reset()
        PT, info, _aud = E.run_fold(XX[dev], yy[dev], zz[dev], gg[dev], XX[te], zz[te],
                                    seed * 100 + kf)
        for p in E.PROGS:
            if p in PT: preds[p][te] = PT[p]
        _drec, _dsum = E.diag_take()
        # **必须是 kf**：`k` 是上一个循环（给 lab 赋外层折号）的残留变量，
        # 用它会把全部诊断行标成最后一折。结果与预测用的是另一套正确变量，
        # 故这是诊断追溯错误，不是预测错误。
        _diag_rows.extend([dict(fold=kf, **r) for r in _drec])
        _diag_sums.append(dict(fold=kf, **_dsum))
        infos.append(dict(fold=kf, n_te=int(len(te)), outer_hash=outer_hash,
                          cache_key=key, **info))
    base = preds["SHARED+"]; avail = ~np.isnan(base)
    orc = E.oracle_lower_bound(preds, yy, zz, avail)
    # 逐就诊预测落盘（#28；本地保留，交付不含身份信息）
    pr = pd.DataFrame({"unit_id": X.index.to_numpy()[idx], "group_id": pid,
                       "hospital_id": meta.hospital_id.astype(str).to_numpy()[idx],
                       "outcome": oc, "phenotype": ph, "seed": seed, "fold": lab,
                       "y": yy, "subgroup": np.where(zz, "A", "B")})
    for p in E.PROGS: pr[p] = preds[p]
    pr[E.ORACLE] = orc
    pr.to_parquet(PRED / f"{tag}__preds.parquet", index=False)
    rows = []
    for p in E.PROGS + [E.ORACLE]:
        q = orc if p == E.ORACLE else preds[p]
        ok = ~np.isnan(q) & avail
        if ok.sum() == 0: continue
        r = dict(dataset=db, outcome=oc, phenotype=ph, seed=seed, 程序=p, n=int(ok.sum()),
                 A轨n=int((zz & ok).sum()), B轨n=int((~zz & ok).sum()),
                 A轨事件=int(yy[zz & ok].sum()), B轨事件=int(yy[~zz & ok].sum()),
                 n_groups=int(len(np.unique(gg))), outer_hash=outer_hash)
        for g, gmk in (("A", zz & ok), ("B", ~zz & ok)):
            rb = float(C.brier(yy[gmk], base[gmk]).mean()); rp = float(C.brier(yy[gmk], q[gmk]).mean())
            r[f"R_{g}"] = rp; r[f"Δ_{g}"] = rb - rp
        r["R_equal"] = .5 * (r["R_A"] + r["R_B"])
        r["Rbase_equal"] = .5 * (float(C.brier(yy[zz & ok], base[zz & ok]).mean())
                                 + float(C.brier(yy[~zz & ok], base[~zz & ok]).mean()))
        r["Δ_equal"] = .5 * (r["Δ_A"] + r["Δ_B"])
        r["相对Δ_equal"] = r["Δ_equal"] / r["Rbase_equal"] if r["Rbase_equal"] > 0 else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_parquet(f, index=False)
    pd.DataFrame(infos).to_parquet(OUTD / f"{tag}__info.parquet", index=False)
    # 真实拟合诊断：全覆盖（outer/inner/calibration 的每一次拟合），不是抽样，不是字面值
    DGD = OUTD / "diag"; DGD.mkdir(exist_ok=True)
    if _diag_rows:
        pd.DataFrame(_diag_rows).to_parquet(DGD / f"{tag}__fits.parquet", index=False)
    if _diag_sums:
        sm = pd.DataFrame(_diag_sums)
        sm["thread_state"] = json.dumps(E.thread_state_now(), ensure_ascii=False)
        sm.to_parquet(DGD / f"{tag}__diagsum.parquet", index=False)
    return tag



def one(db, oc, ph, seed):
    """单元级包装：失败就**如实记录**并继续，不让一个单元带崩整轮。

    纪律：这不是折级的 except-pass。折内任何拟合失败仍然立刻抛出；
    这里只在**单元边界**把失败转成 manifest 里的一条 FAILED 记录，
    使 72 个单元的尝试与结果都能被核对（GPT 第 39 轮：不用抽查代替全量状态记录）。
    """
    import traceback as _tb
    args = (db, oc, ph, seed,)
    tag = "__".join(str(a) for a in args)
    try:
        return _one_impl(db, oc, ph, seed)
    except Exception as _e:
        FD = OUTD / "failed"; FD.mkdir(parents=True, exist_ok=True)
        (FD / f"{tag}.txt").write_text(
            f"{type(_e).__name__}: {_e}\n\n{_tb.format_exc()}", encoding="utf-8")
        print(f"  !! 单元失败 {tag}: {type(_e).__name__}: {str(_e)[:160]}", flush=True)
        return "FAIL"

def main(db, jobs):
    jl = [(db, oc, ph, s) for oc in OUTCOMES for ph in PHENOS for s in SEEDS]
    # 结果文件名**含缓存键**，过滤必须同样带键。
    # 原写法漏了键 ⇒ todo 永远是全部单元，只是派发后在 worker 里空转一次。
    def _done(j):
        return (OUTD / f"{j[0]}__{j[1]}__{j[2]}__{j[3]}__{cache_key(*j)}.parquet").exists()
    todo = [j for j in jl if not _done(j)]
    print(f"{db}：{len(jl)} 个单元，待跑 {len(todo)}，并行 {jobs}", flush=True)
    t0 = time.time()
    # 每个单元跑完就回收 worker 进程。
    # R38.1 实测：joblib/loky 的 worker 跨单元累积内存，eICU 下涨到 2.4GB/个，
    # 把 15.8GB 挤满后 Windows 换页，CPU 利用率从 ~100% 掉到 12%。
    # joblib 1.5.3 的 Parallel **没有** max_tasks_per_child（那是 loky 执行器的参数），
    # 故改用标准库 ProcessPoolExecutor（Python 3.11+ 支持该参数），每单元起干净进程。
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        futs = {ex.submit(one, *j): j for j in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            try:
                out.append(fu.result())
            except Exception as _e:
                print(f"  !! 提交层异常 {j}: {type(_e).__name__}: {str(_e)[:160]}", flush=True)
                out.append(None)
            print(f"  [{i}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
    # 汇总以**实际产物**为准：返回值里 SKIP_DONE/SKIP_TOO_SMALL/FAIL 含义不同，
    # 不能一律当成失败（原写法把「已完成」和「样本不足」也计入失败数）。
    nres = sum(1 for j in jl if _done(j))
    nfail = len(list((OUTD / "failed").glob("*.txt"))) if (OUTD / "failed").exists() else 0
    from collections import Counter
    cnt = Counter(str(o) if isinstance(o, str) else ("OK" if o else "None") for o in out)
    print(f"{db}：结果文件 {nres}/{len(jl)}；失败记录 {nfail}；"
          f"返回值分布 {dict(cnt)}  {time.time()-t0:.0f}s", flush=True)
    if nfail:
        print(f"  失败详情见 {OUTD/'failed'}", flush=True)
    if nres + nfail < len(jl):
        print(f"  !! 既无结果也无失败记录的单元 {len(jl)-nres-nfail} 个（样本不足或被跳过）",
              flush=True)


if __name__ == "__main__":
    db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "mimic"
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    main(db, jobs)
