# -*- coding: utf-8 -*-
r"""执行路径资格审计（GPT 第 47 轮 §4）：用真实引擎的干跑模式走遍正式批次将执行的全部拟合路径。

覆盖
  main  ：24 任务 × 3 seed × 5 外层折 —— develop(D) + final_predict（全部臂 raw/cal）
  cert  ：同上 —— develop(F, 合并全部 + Hfull + HTFS) + P*_F 与 HTFS_F 在 C 与外层测试上的预测
  LC    ：24 任务 × 3 seed × 5 外层折 × {25%,50%,75%} —— develop(子集, 合并全部 + Hfull + HTFS) + P*/Hfull/HTFS 测试预测
干跑：fit_booster 做全部检查与划分（含专属轨自己的早停留出、合并早停留出的两轨、校准折、最终拟合），记录计划、不训练。

分类（预先规定）
  不可执行：任何 FoldFailure（空集合、单类别、组数不足无法划分、校准不可行）→ 记录任务/seed/外层折/分支/角色/信息
  可执行但不稳定：存在某次拟合的早停留出，专属轨事件或非事件 < 10，或合并留出任一轨事件或非事件 < 10
  学习曲线点可评价：可执行 且 训练子集每轨事件与非事件 ≥ 100（工程筛查）
不搜索替代种子、不删除候选、不把失败记为零收益。

输出：23_R42_正式主实验/pathaudit/R42_PATHAUDIT.xlsx 与 .json
"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

OUT = HERE.parent / "23_R42_正式主实验" / "pathaudit"
UNSTABLE = 10
LC_MIN = 100


def task_arrays(db, oc, ph):
    import pgss_r38_bench as BN, pgss_xgb_engine as EN
    X, P, OK, meta = BN.load(db)
    e = meta.eligible.fillna(False).to_numpy(bool)
    idx = np.flatnonzero(e & OK[ph].to_numpy(bool))
    y = meta[oc].fillna(False).to_numpy(bool).astype(int)[idx]
    z = P[ph].to_numpy(bool)[idx]
    pid = meta.group_id.astype(str).to_numpy()[idx]
    gs = pid if db == "mimic" else BN.joint_groups(pid, meta.hospital_id.astype(str).to_numpy()[idx])
    A = dict(X=X.to_numpy(float)[idx], y=y, z=z, g=EN.group_codes(gs), uid=X.index.to_numpy()[idx].astype(np.int64))
    return A, list(X.columns), gs


def summarize(plans):
    """从拟合计划中统计最小留出事件/非事件，判定是否不稳定。"""
    if not plans:
        return dict(拟合计划数=0)
    P = pd.DataFrame(plans)
    trk = P[~P.pooled]; pol = P[P.pooled]
    mins = {}
    if len(trk):
        mins["专属留出事件最小"] = int(trk.hold_events.min()); mins["专属留出非事件最小"] = int(trk.hold_nonevents.min())
    if len(pol):
        mins["合并留出轨事件最小"] = int(min(pol.hold_evA.min(), pol.hold_evB.min()))
        mins["合并留出轨非事件最小"] = int(min(pol.hold_neA.min(), pol.hold_neB.min()))
    unstable = any(v < UNSTABLE for v in mins.values())
    worst = None
    if unstable:
        sc = []
        for _, r in P.iterrows():
            v = min(r.hold_events, r.hold_nonevents) if not r.pooled else min(r.hold_evA, r.hold_evB, r.hold_neA, r.hold_neB)
            sc.append(v)
        worst = P.iloc[int(np.argmin(sc))].role
    return dict(拟合计划数=len(P), 去重训练集合=int(P.plan_key.nunique()), 最小训练组数=int(P.groups.min()),
                **mins, 不稳定=unstable, 最不稳定角色=worst)


def audit_task(args):
    db, oc, ph = args
    import pgss_r38_bench as BN, pgss_r36_engine as E, pgss_xgb_engine as EN
    EN.DRYRUN = True
    A, names, gs = task_arrays(db, oc, ph)
    rows, fails = [], []
    for seed in BN.SEEDS:
        folds = E.group_split(A["X"], A["y"], A["z"], gs, BN.K_OUTER, seed)
        for kf, (dev, te) in enumerate(folds):
            D, Te = EN.sub(A, dev), EN.sub(A, te)
            lay = EN.task_layout(D["X"], D["z"], names)
            units = [("main", None)] + [("cert", None)] + [("LC", fr) for fr in EN.LC_FRACS]
            for br, fr in units:
                EN.set_namespace(db=db, oc=oc, ph=ph, seed=seed, fold=kf, branch=br, frac=fr, mode="dryrun")
                base = dict(库=db, 结局=oc, 表型=ph, seed=seed, 外层折=kf, 分支=br, 比例=fr)
                extra = {}
                try:
                    if br == "main":
                        r = EN.develop(D, lay, seed * 100 + kf)
                        EN.final_predict(r["specs"], D, Te, lay)
                    elif br == "cert":
                        keep, hold = EN.cert_split(D["g"], seed * 100 + kf + 50_000)
                        F, C = EN.sub(D, keep), EN.sub(D, hold)
                        extra = dict(F_n=len(keep), C_n=len(hold))
                        r = EN.develop(F, lay, seed * 100 + kf + 70_000, arms_tracked=("Hfull", "HTFS"))
                        for nm in (r["pstar"], "HTFS"):
                            EN.program_predict(r["specs"][nm], F, C, lay); EN.program_predict(r["specs"][nm], F, Te, lay)
                    else:
                        si = EN.nested_fraction(D["g"], fr, seed * 10 + kf)
                        S = EN.sub(D, si)
                        ev = dict(A事件=int(S["y"][S["z"]].sum()), A非事件=int((1 - S["y"][S["z"]]).sum()),
                                  B事件=int(S["y"][~S["z"]].sum()), B非事件=int((1 - S["y"][~S["z"]]).sum()))
                        extra = dict(子集n=len(si), 子集组数=int(len(np.unique(S["g"]))), **ev,
                                     达到LC事件门槛=bool(min(ev.values()) >= LC_MIN))
                        r = EN.develop(S, lay, seed * 100 + kf, arms_tracked=("Hfull", "HTFS"))
                        EN.final_predict(r["specs"], S, Te, lay, names=[r["pstar"], "Hfull", "HTFS"])
                    rows.append(dict(base, 可执行=True, **extra, **summarize(EN.FITLOG)))
                except EN.FoldFailure as ex:
                    last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
                    rows.append(dict(base, 可执行=False, **extra, 失败信息=str(ex), 失败前最后角色=last))
                    fails.append(dict(base, 信息=str(ex), 失败前最后角色=last))
    return rows, fails


def main(jobs=16):
    from concurrent.futures import ProcessPoolExecutor
    import pgss_r38_bench as BN
    OUT.mkdir(parents=True, exist_ok=True)
    jl = [(db, oc, ph) for db in ("eicu", "mimic") for oc in BN.OUTCOMES for ph in BN.PHENOS]
    t0 = time.time(); rows, fails = [], []
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for (r, f), a in zip(ex.map(audit_task, jl), jl):
            rows += r; fails += f
            print(f"  {a}  {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); Fl = pd.DataFrame(fails)
    R["LC可评价"] = np.where(R.分支 == "LC", R.可执行 & R.get("达到LC事件门槛", False).fillna(False).astype(bool), np.nan)
    S = (R.groupby(["库", "分支", "比例"], dropna=False)
          .agg(单元=("可执行", "size"), 可执行=("可执行", "sum"), 不稳定=("不稳定", lambda s: int(s.fillna(False).sum())))
          .reset_index())
    lc = R[R.分支 == "LC"]
    S2 = lc.groupby(["库", "比例"]).agg(点=("可执行", "size"), 可执行=("可执行", "sum"),
                                        可评价=("LC可评价", lambda s: int(s.fillna(0).sum()))).reset_index()
    summ = dict(单元=len(R), 可执行=int(R.可执行.sum()), 不可执行=int((~R.可执行).sum()),
                主分支与认证分支_不可执行=int((~R[R.分支 != "LC"].可执行).sum()),
                主分支与认证分支_不稳定=int(R[R.分支 != "LC"].不稳定.fillna(False).sum()),
                LC点=len(lc), LC可评价=int(lc.LC可评价.fillna(0).sum()), 秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R42_PATHAUDIT.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).to_excel(xw, sheet_name="0_汇总", index=False)
        S.to_excel(xw, sheet_name="1_按库分支", index=False)
        S2.to_excel(xw, sheet_name="2_学习曲线可评价", index=False)
        (Fl if len(Fl) else pd.DataFrame([{"说明": "无不可执行单元"}])).to_excel(xw, sheet_name="3_不可执行定位", index=False)
        R[R.不稳定.fillna(False).astype(bool)].to_excel(xw, sheet_name="4_不稳定单元", index=False)
        R.to_excel(xw, sheet_name="5_全部单元", index=False)
    R.to_parquet(OUT / "R42_PATHAUDIT_units.parquet", index=False)
    (OUT / "R42_PATHAUDIT.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "_DONE").write_text("ok", encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
