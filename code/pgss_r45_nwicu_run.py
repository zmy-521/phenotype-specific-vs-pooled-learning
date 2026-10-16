# -*- coding: utf-8 -*-
r"""R45 NWICU 独立程序复现：运行单元（= R42 main 分支的同一学习程序，只换数据源；GPT 第 49 轮 §6.4）。

一个单元 = 结局 × 表型 × seed × 外层折。外层划分 = R42 同一函数（pgss_r36_engine.group_split，事件均衡，K=5，按患者分组），
seed 20261003–05；develop(D) → 全部 11 臂在 D 上冻结拟合 → 外层测试 raw/cal 预测。引擎 xgb-1.0 不改任何配置。
FoldFailure → 写 _FAILED，不换种子、不删候选、不记为零收益；按实际可执行性记录不完整任务。
数据：27_R45_外部复现与机制核验/nwicu/data（pgss_r45_nwicu_build.py；X 列序与 MIMIC 特征池相同）。
输出：27_R45_外部复现与机制核验/nwicu/runs/<tag>/（preds_test 含 unit_id=stay_id，只留本地）
用法：python pgss_r45_nwicu_run.py run <jobs> | audit <jobs> | one <oc> <ph> <seed> <fold>
"""
import sys, os, json, pathlib, time, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

NW = HERE.parent / "27_R45_外部复现与机制核验" / "nwicu"
DATA = NW / "data"; RUNS = NW / "runs"
OUTCOMES = ["OUT_HOSP_MORT", "OUT_LOS72"]
PHENOS = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
SEEDS = [20261003, 20261004, 20261005]
K_OUTER = 5
CODE_FILES = ["pgss_r45_nwicu_run.py", "pgss_r45_nwicu_build.py", "pgss_xgb_engine.py", "pgss_r36_engine.py"]


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def data_fp():
    return json.loads((DATA / "COHORT.json").read_text(encoding="utf-8"))["指纹"]


def tag_of(oc, ph, seed, fold):
    return f"nwicu__{oc}__{ph}__{seed}__{fold}__main"


def task_arrays(oc, ph):
    import pgss_xgb_engine as EN
    X = pd.read_parquet(DATA / "X.parquet"); P = pd.read_parquet(DATA / "P.parquet")
    OK = pd.read_parquet(DATA / "OK.parquet"); meta = pd.read_parquet(DATA / "meta.parquet")
    assert X.index.equals(P.index) and X.index.equals(OK.index) and X.index.equals(meta.index)
    idx = np.flatnonzero(meta.eligible.to_numpy(bool) & OK[ph].to_numpy(bool))
    gs = meta.group_id.astype(str).to_numpy()[idx]
    A = dict(X=X.to_numpy(float)[idx], y=meta[oc].to_numpy(bool).astype(int)[idx], z=P[ph].to_numpy(bool)[idx],
             g=EN.group_codes(gs), uid=X.index.to_numpy()[idx].astype(np.int64))
    return A, list(X.columns), gs


def fsha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def run_unit(oc, ph, seed, fold):
    import psutil
    import pgss_r36_engine as E, pgss_xgb_engine as EN
    tag = tag_of(oc, ph, seed, fold)
    od = RUNS / tag; od.mkdir(parents=True, exist_ok=True)
    csha = code_sha()
    if (od / "_DONE").exists():
        return "SKIP_DONE"
    (od / "_FAILED").unlink(missing_ok=True)
    t0 = time.time(); proc = psutil.Process()
    A, names, gs = task_arrays(oc, ph)
    folds = E.group_split(A["X"], A["y"], A["z"], gs, K_OUTER, seed)
    dev, te = folds[fold]
    D, Te = EN.sub(A, dev), EN.sub(A, te)
    lay = EN.task_layout(D["X"], D["z"], names)
    fp = data_fp()
    EN.set_namespace(db="nwicu", oc=oc, ph=ph, seed=seed, fold=fold, branch="main", data_fp=fp, code=csha)
    meta = dict(tag=tag, db="nwicu", oc=oc, ph=ph, seed=seed, fold=fold, branch="main", engine_version=EN.ENGINE_VERSION,
                code_sha=csha, data_fp=fp, z_name=lay["z_name"], xgb_nthread=EN.PARAM_BASE["nthread"],
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                n_dev=len(dev), n_test=len(te), events_dev=int(D["y"].sum()), nA_dev=int(D["z"].sum()),
                eventsA_dev=int(D["y"][D["z"]].sum()))
    out = pd.DataFrame({"unit_id": Te["uid"], "y": Te["y"], "z": Te["z"]})
    try:
        r = EN.develop(D, lay, seed * 100 + fold)
        fpred = EN.final_predict(r["specs"], D, Te, lay)
        for k, v in fpred.items():
            out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
        meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fpred.items()})
    except EN.FoldFailure as ex:
        last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
        (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last, trace=traceback.format_exc()[-1500:]),
                                               ensure_ascii=False, indent=1), encoding="utf-8")
        return "FAILED"
    if EN.DRYRUN:
        (od / "_DONE").write_text("dryrun", encoding="utf-8")
        return "OK"
    out.to_parquet(od / "preds_test.parquet", index=False)
    (od / "specs.json").write_text(json.dumps(dict(specs=r["specs"], inner=r["inner"], pstar=r["pstar"],
                                                   inner_folds=r["inner_folds"]), ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(r["rows"]).to_parquet(od / "inner_scores.parquet", index=False)
    FL = pd.DataFrame(EN.FITLOG); FL.to_parquet(od / "fitlog.parquet", index=False)
    pd.DataFrame(EN.CALLOG).to_parquet(od / "callog.parquet", index=False)
    meta.update(sec_wall=round(time.time() - t0, 1), n_fits=len(FL),
                peak_rss_mb=round(proc.memory_info().rss / 2**20, 1), requests=dict(EN.REQLOG))
    meta["artifacts"] = {f.name: fsha(f) for f in sorted(od.glob("*.parquet"))}
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (od / "_DONE").write_text(str(meta["sec_wall"]), encoding="utf-8")
    return "OK"


def _entry(args):
    try:
        return args, run_unit(*args)
    except Exception as e:
        od = RUNS / tag_of(*args); od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"非预期异常 {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", encoding="utf-8")
        return args, "ERROR"


def units():
    order = ["女性", "年龄≥65", "eGFR降低", "贫血", "低白蛋白", "肥胖"]      # 人数多的在前
    return [(oc, ph, s, f) for ph in order for oc in OUTCOMES for s in SEEDS for f in range(K_OUTER)]


def run(jobs):
    from concurrent.futures import ProcessPoolExecutor
    RUNS.mkdir(parents=True, exist_ok=True)
    jl = [a for a in units() if not (RUNS / tag_of(*a) / "_DONE").exists()]
    ok = fail = 0; t0 = time.time(); st = NW / "_STATUS.json"
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res in ("FAILED", "ERROR")
            st.write_text(json.dumps(dict(总数=len(units()), 本次待跑=len(jl), 完成=ok, 失败或不可执行=fail,
                                          已用分钟=round((time.time() - t0) / 60, 1), 更新=time.strftime("%m-%d %H:%M:%S")),
                                     ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    (NW / "_RUNS_DONE").write_text(f"ok={ok} fail={fail}", encoding="utf-8")


def _audit_entry(args):
    """执行路径审计（干跑）：真实引擎 DRYRUN 模式走完全部划分与检查，不训练、不产生预测；输出写在 pathaudit_runs/。"""
    global RUNS
    import pgss_xgb_engine as EN
    EN.DRYRUN = True
    RUNS = NW / "pathaudit_runs"
    try:
        res = run_unit(*args)
    except Exception as e:
        res = f"ERROR {type(e).__name__}: {e}"
    od = RUNS / tag_of(*args)
    info = ""
    if (od / "_FAILED").exists():
        try:
            j = json.loads((od / "_FAILED").read_text(encoding="utf-8")); info = f"{j.get('error')} @ {j.get('last_role')}"
        except Exception:
            info = (od / "_FAILED").read_text(encoding="utf-8")[:200]
    return dict(结局=args[0], 表型=args[1], seed=args[2], 外层折=args[3], 结果=res, 信息=info)


def audit(jobs):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        rows = list(ex.map(_audit_entry, units()))
    A = pd.DataFrame(rows)
    A.to_csv(NW / "R45_NWICU_PATHAUDIT.csv", index=False, encoding="utf-8-sig")
    s = A.groupby(["结局", "表型"]).结果.apply(lambda v: f"{(v == 'OK').sum()}/{len(v)}").reset_index()
    (NW / "R45_NWICU_PATHAUDIT.json").write_text(json.dumps(dict(可执行=int((A.结果 == "OK").sum()), 总数=len(A),
                                                                按任务=s.to_dict("records"), code_sha=code_sha(), data_fp=data_fp()),
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    print(s.to_string()); print(A[A.结果 != "OK"].to_string())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]))
    elif sys.argv[1] == "audit":
        audit(int(sys.argv[2]))
    elif sys.argv[1] == "one":
        print(run_unit(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])))
