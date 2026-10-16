# -*- coding: utf-8 -*-
r"""R45 NHANES 目标一致性追加对照（GPT 第 49 轮 §4.3–4.4）：合并程序的开发目标从两轨等权改为指定目标，贯穿全部合并开发环节。

目标（R_q = q·R_A + (1−q)·R_B）
  bal ：q = 0.5（= 冻结引擎 xgb-1.0 原样；只用于等价测试）
  pop ：按记录等权（= 评分集中的样本比例；NHANES 11 个已纳入任务）
  B   ：q = 0（只用 B 轨评分指导开发；仍用两轨训练数据；只在 O_MORT10×低白蛋白）

随目标改变的环节（在运行时替换 pgss_xgb_engine 的模块级函数名，**不改冻结文件**；适配器有独立版本号与缓存命名空间）
  1. 早停评分权重 es_metric_weights（合并臂拟合、全局排序模型 rank|all —— 后者决定 SZ-GFS 面板）
  2. 合并臂内层评分：S0/SZ 的 θ、SZ-GFS 的 k、各合并臂 raw/cal 开关、P* 选择
  3. HTFS 的 raw/cal 开关：参数与面板冻结（读 R44 该单元 specs.json），只在已有两版本中按目标的内层 OOF 分数重选
不变：训练权重候选 {none, bal}、候选族与网格、并列规则、外层划分与内层划分、轨内排序与 kA/kB
  （专属臂按本轨开发、与目标无关；SZ-union 的轨面板 = 同一训练部分内 track_rank 的前 kA/kB，kA/kB 取 R44 该单元的冻结值）。
早停留出缺某一轨时照原引擎抛 FoldFailure（三个目标的可执行集合相同）。

用法：python pgss_r45_target.py one <target> <oc> <ph> <seed> <fold> | run <target> <jobs> | equiv
输出：27_R45_外部复现与机制核验/target/runs/<tag>/（preds_test 含 unit_id=SEQN，只留本地）
"""
import sys, os, json, pathlib, time, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ADAPTER_VERSION = "r45-target-1.0"
R45 = HERE.parent / "27_R45_外部复现与机制核验"
OUT = R45 / "target"; RUNS = OUT / "runs"
TARGETS = ("bal", "pop", "B")
B_TASKS = [("O_MORT10", "低白蛋白")]
CODE_FILES = ["pgss_r45_target.py", "pgss_r44_nhanes_run.py", "pgss_r44_nhanes_cohort.py", "pgss_xgb_engine.py",
              "pgss_r36_engine.py"]

TARGET = None
CALLS = {"es_pooled": 0, "es_track": 0, "score": 0}
_ORIG = {}


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


# ── 目标化的两个函数 ─────────────────────────────────────────────
def es_weights_target(z, pooled):
    """早停评分权重。专属臂（pooled=False）与原引擎完全相同；合并臂按目标。"""
    import pgss_xgb_engine as EN
    z = np.asarray(z, bool)
    if not pooled:
        CALLS["es_track"] += 1
        return _ORIG["es"](z, pooled)
    CALLS["es_pooled"] += 1
    if TARGET == "bal":
        return _ORIG["es"](z, pooled)
    nA, nB = int(z.sum()), int((~z).sum())
    if nA == 0 or nB == 0:
        raise EN.FoldFailure("早停留出缺少某一轨（与原引擎相同的可执行条件）")
    if TARGET == "pop":
        return np.ones(len(z))
    if TARGET == "B":
        return np.where(z, 0.0, 1.0 / nB)
    raise ValueError(TARGET)


def score_target(y, p, z):
    """内层评分 R_q。bal 直接调用原引擎 brier_bal（等价测试要求逐位相同）。"""
    import pgss_xgb_engine as EN
    CALLS["score"] += 1
    z = np.asarray(z, bool)
    if TARGET == "bal":
        return EN.brier_bal(y, p, z)
    if TARGET == "pop":
        return float(np.mean((y - p) ** 2))
    if TARGET == "B":
        return EN.brier_track(y, p, z, "B")
    raise ValueError(TARGET)


def activate(target):
    """设置目标并替换引擎的早停权重函数（幂等）。"""
    global TARGET
    import pgss_xgb_engine as EN
    if target not in TARGETS:
        raise ValueError(target)
    if "es" not in _ORIG:
        _ORIG["es"] = EN.es_metric_weights
    EN.es_metric_weights = es_weights_target
    TARGET = target
    for k in CALLS:
        CALLS[k] = 0


def deactivate():
    import pgss_xgb_engine as EN
    if "es" in _ORIG:
        EN.es_metric_weights = _ORIG["es"]


# ── 合并臂开发（= EN.develop 的合并部分，逐行对应；评分换成 score_target）──────
def develop_pooled(D, lay, seed, kk):
    import pgss_xgb_engine as EN
    folds = EN.group_split(D["g"], EN.INNER_FOLDS, seed)
    rows = []

    def oof(spec, tracks=("A", "B")):
        o = np.full(len(D["y"]), np.nan)
        for tr, te in folds:
            o[te] = EN.program_predict(spec, EN.sub(D, tr), EN.sub(D, te), lay, tracks)
        return o

    def sc_pooled(spec):
        s = score_target(D["y"], oof(spec), D["z"])
        rows.append(dict(scope="pooled", spec=json.dumps(spec, ensure_ascii=False, sort_keys=True), score=s))
        return s

    specs = {}
    k_gfs = {}
    for w in EN.WEIGHTS:
        tS0 = min(EN.GRID_POOLED, key=lambda t: sc_pooled(dict(arch="S0", w=w, theta=t)))
        specs[f"S0|{w}"] = dict(arch="S0", w=w, theta=tS0)
        if lay["z_is_col"] is not None:
            tSZ = tS0
            specs[f"SZ|{w}"] = dict(arch="S0", w=w, theta=tS0)
        else:
            tSZ = min(EN.GRID_POOLED, key=lambda t: sc_pooled(dict(arch="SZ", w=w, theta=t)))
            specs[f"SZ|{w}"] = dict(arch="SZ", w=w, theta=tSZ)
        sz_spec = specs[f"SZ|{w}"]
        cand = {}
        for k in EN.K_GRID:
            s = sz_spec if k == 23 else dict(arch="SZ-GFS", w=w, theta=tSZ, k=k)
            cand[k] = sc_pooled(s)
        kg = min(EN.K_GRID, key=lambda k: (cand[k], -k))
        k_gfs[w] = kg
        specs[f"SZ-GFS|{w}"] = dict(sz_spec) if kg == 23 else dict(arch="SZ-GFS", w=w, theta=tSZ, k=kg)
        specs[f"SZ-union|{w}"] = dict(arch="SZ-union", w=w, theta=tSZ, kA=kk["A"], kB=kk["B"])
    final, inner = {}, {}
    for name, s in specs.items():
        r = sc_pooled(dict(s, cal=False)); c = sc_pooled(dict(s, cal=True))
        final[name] = dict(s, cal=bool(c < r)); inner[name] = min(r, c)
    order = [f"{a}|{w}" for a in EN.POOLED for w in EN.WEIGHTS if f"{a}|{w}" in final]
    pstar = min(order, key=lambda n: (inner[n], order.index(n)))
    return dict(specs=final, inner=inner, pstar=pstar, rows=rows, folds=folds,
                inner_folds=[(len(tr), len(te)) for tr, te in folds])


def htfs_switch(D, lay, folds, htfs_frozen):
    """冻结 HTFS（参数、面板来自 R44）在同一内层划分上的 raw 与 cal OOF；返回逐轨、bal、pop、目标分数与按目标选的版本。"""
    import pgss_xgb_engine as EN
    sc = {}
    for ver, cal in (("raw", False), ("cal", True)):
        spec = dict(htfs_frozen, cal=cal)
        o = np.full(len(D["y"]), np.nan)
        for tr, te in folds:
            o[te] = EN.program_predict(spec, EN.sub(D, tr), EN.sub(D, te), lay)
        y, z = D["y"], D["z"]
        sc[ver] = dict(A=EN.brier_track(y, o, z, "A"), B=EN.brier_track(y, o, z, "B"), bal=EN.brier_bal(y, o, z),
                       pop=float(np.mean((y - o) ** 2)), target=score_target(y, o, z))
    return sc, ("cal" if sc["cal"]["target"] < sc["raw"]["target"] else "raw")


# ── 单元 ──────────────────────────────────────────────────────
def tag_of(target, oc, ph, seed, fold):
    return f"nhanes__{oc}__{ph}__{seed}__{fold}__target-{target}"


def run_unit(target, oc, ph, seed, fold, runs=None):
    import psutil
    import pgss_r36_engine as E, pgss_xgb_engine as EN, pgss_r44_nhanes_run as NR
    runs = RUNS if runs is None else runs
    tag = tag_of(target, oc, ph, seed, fold)
    od = runs / tag; od.mkdir(parents=True, exist_ok=True)
    csha = code_sha()
    if (od / "_DONE").exists():
        return "SKIP_DONE"
    (od / "_FAILED").unlink(missing_ok=True)
    t0 = time.time(); proc = psutil.Process()
    src = NR.RUNS / NR.tag_of(oc, ph, seed, fold)
    sp44 = json.loads((src / "specs.json").read_text(encoding="utf-8"))
    h = {k: v for k, v in sp44["specs"]["HTFS"].items() if k != "cal"}
    A, names, gs = NR.task_arrays(oc, ph)
    folds = E.group_split(A["X"], A["y"], A["z"], gs, NR.K_OUTER, seed)
    dev, te = folds[fold]
    D, Te = EN.sub(A, dev), EN.sub(A, te)
    lay = EN.task_layout(D["X"], D["z"], names)
    fp = NR.data_fp()
    activate(target)
    EN.set_namespace(db="nhanes", oc=oc, ph=ph, seed=seed, fold=fold, branch=f"target-{target}", adapter=ADAPTER_VERSION,
                     data_fp=fp, code=csha)
    meta = dict(tag=tag, target=target, adapter_version=ADAPTER_VERSION, db="nhanes", oc=oc, ph=ph, seed=seed, fold=fold,
                engine_version=EN.ENGINE_VERSION, code_sha=csha, data_fp=fp, z_name=lay["z_name"],
                r44_specs_sha=hashlib.sha256((src / "specs.json").read_bytes()).hexdigest()[:16],
                htfs_frozen=h, n_dev=len(dev), n_test=len(te), nA_dev=int(D["z"].sum()), events_dev=int(D["y"].sum()),
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    out = pd.DataFrame({"unit_id": Te["uid"], "y": Te["y"], "z": Te["z"]})
    try:
        r = develop_pooled(D, lay, seed * 100 + fold, {"A": h["kA"], "B": h["kB"]})
        hs, hchosen = htfs_switch(D, lay, r["folds"], h)
        fpred = EN.final_predict(r["specs"], D, Te, lay)
        for k, v in fpred.items():
            out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
        meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fpred.items()}, htfs_chosen_target=hchosen,
                    calls=dict(CALLS))
    except EN.FoldFailure as ex:
        last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
        (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last, trace=traceback.format_exc()[-1500:]),
                                               ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return "FAILED"
    finally:
        deactivate()
    out.to_parquet(od / "preds_test.parquet", index=False)
    (od / "specs.json").write_text(json.dumps(dict(target=target, specs=r["specs"], inner=r["inner"], pstar=r["pstar"],
                                                   inner_folds=r["inner_folds"], htfs_frozen=h, htfs_inner=hs,
                                                   htfs_chosen_target=hchosen), ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    pd.DataFrame(r["rows"]).to_parquet(od / "inner_scores.parquet", index=False)
    FL = pd.DataFrame(EN.FITLOG); FL.to_parquet(od / "fitlog.parquet", index=False)
    pd.DataFrame(EN.CALLOG).to_parquet(od / "callog.parquet", index=False)
    meta.update(sec_wall=round(time.time() - t0, 1), n_fits=len(FL),
                peak_rss_mb=round(proc.memory_info().rss / 2**20, 1), requests=dict(EN.REQLOG))
    meta["artifacts"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(od.glob("*.parquet"))}
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (od / "_DONE").write_text(str(meta["sec_wall"]), encoding="utf-8")
    return "OK"


def units(target):
    import pgss_r44_nhanes_run as NR
    inc = json.loads((OUT / "INCLUDED_TASKS.json").read_text(encoding="utf-8"))
    tasks = [tuple(t) for t in inc["纳入任务"]] if target == "pop" else B_TASKS if target == "B" else []
    return [(target, oc, ph, s, f) for (oc, ph) in tasks for s in NR.SEEDS for f in range(NR.K_OUTER)]


def _entry(args):
    try:
        return args, run_unit(*args)
    except Exception as e:
        od = RUNS / tag_of(*args); od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"非预期异常 {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", encoding="utf-8")
        return args, "ERROR"


def run(targets, jobs):
    from concurrent.futures import ProcessPoolExecutor
    RUNS.mkdir(parents=True, exist_ok=True)
    allu = [u for t in targets for u in units(t)]
    jl = [a for a in allu if not (RUNS / tag_of(*a) / "_DONE").exists()]
    ok = fail = 0; t0 = time.time(); st = OUT / "_STATUS.json"
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res in ("FAILED", "ERROR")
            st.write_text(json.dumps(dict(总数=len(allu), 本次待跑=len(jl), 完成=ok, 失败或不可执行=fail,
                                          已用分钟=round((time.time() - t0) / 60, 1), 更新=time.strftime("%m-%d %H:%M:%S")),
                                     ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    return ok, fail


# ── q = 0.5 等价测试：适配器在 bal 目标下必须逐位复现 R44 冻结单元 ──────────
EQUIV_UNITS = [("O_MORT10", "低白蛋白", 20261003, 0), ("O_CVD10", "女性", 20261004, 2)]


def equiv_compare(new_dir, old_dir):
    """返回不一致项列表（空 = 逐位一致）。比较：合并臂规格、内层分数、P*、合并臂测试预测（raw/cal）、HTFS 内层 bal 分数与原选择。"""
    bad = []
    n = json.loads((new_dir / "specs.json").read_text(encoding="utf-8"))
    o = json.loads((old_dir / "specs.json").read_text(encoding="utf-8"))
    pooled = [k for k in o["specs"] if k.split("|")[0] in ("S0", "SZ", "SZ-GFS", "SZ-union")]
    if sorted(pooled) != sorted(n["specs"]):
        bad.append(f"合并臂集合不同 {sorted(pooled)} vs {sorted(n['specs'])}")
    for k in pooled:
        if n["specs"].get(k) != o["specs"][k]:
            bad.append(f"规格 {k}: {n['specs'].get(k)} vs {o['specs'][k]}")
        if n["inner"].get(k) != o["inner"][k]:
            bad.append(f"内层分数 {k}: {n['inner'].get(k)!r} vs {o['inner'][k]!r}")
    if n["pstar"] != o["pstar"]:
        bad.append(f"P* {n['pstar']} vs {o['pstar']}")
    Pn = pd.read_parquet(new_dir / "preds_test.parquet"); Po = pd.read_parquet(old_dir / "preds_test.parquet")
    if not np.array_equal(Pn.unit_id.to_numpy(), Po.unit_id.to_numpy()):
        bad.append("测试单位或行序不同")
    else:
        for k in pooled:
            for v in ("raw", "cal"):
                c = f"{k}|{v}"
                if not np.array_equal(Pn[c].to_numpy(), Po[c].to_numpy()):
                    bad.append(f"测试预测 {c} 最大差 {np.max(np.abs(Pn[c].to_numpy() - Po[c].to_numpy())):.3e}")
    rows = pd.read_parquet(old_dir / "inner_scores.parquet")
    tb = rows[rows.scope == "tracked-bal"]
    for ver, cal in (("raw", False), ("cal", True)):
        key = json.dumps(dict(n["htfs_frozen"], cal=cal), sort_keys=True)
        hit = tb[tb.spec == key].score.to_numpy()
        if len(hit) != 1 or float(hit[0]) != float(n["htfs_inner"][ver]["bal"]):
            bad.append(f"HTFS 内层 bal 分数 {ver}: {n['htfs_inner'][ver]['bal']!r} vs {hit.tolist()}")
    if n["htfs_chosen_target"] != ("cal" if o["specs"]["HTFS"]["cal"] else "raw"):
        bad.append(f"HTFS 版本选择 {n['htfs_chosen_target']} vs R44")
    return bad


def equiv():
    from concurrent.futures import ProcessPoolExecutor
    import pgss_r44_nhanes_run as NR
    er = OUT / "equiv_runs"; er.mkdir(parents=True, exist_ok=True)
    args = [("bal",) + u for u in EQUIV_UNITS]
    with ProcessPoolExecutor(max_workers=len(args), max_tasks_per_child=1) as ex:
        res = list(ex.map(_equiv_entry, args))
    rep = []
    for a, r in zip(args, res):
        nd = er / tag_of(*a); od = NR.RUNS / NR.tag_of(*a[1:])
        bad = equiv_compare(nd, od) if r == "OK" else [f"运行结果 {r}"]
        m = json.loads((nd / "meta.json").read_text(encoding="utf-8")) if (nd / "meta.json").exists() else {}
        rep.append(dict(单元=tag_of(*a), 运行=r, 不一致项=bad, 一致=not bad, calls=m.get("calls"), 秒=m.get("sec_wall")))
    # 变异对照：把一个合并臂测试预测改动 1e-12、把 P* 改名，比较必须报不一致
    mut = []
    import shutil
    nd = er / tag_of(*args[0]); md = er / "_mutation"
    if md.exists():
        shutil.rmtree(md)
    shutil.copytree(nd, md)
    P = pd.read_parquet(md / "preds_test.parquet"); c = [x for x in P.columns if x.startswith("SZ|")][0]
    P.loc[0, c] = P.loc[0, c] + 1e-12; P.to_parquet(md / "preds_test.parquet", index=False)
    mut.append(dict(变异="合并臂测试预测 +1e-12", 检出=bool(equiv_compare(md, NR.RUNS / NR.tag_of(*args[0][1:])))))
    s = json.loads((md / "specs.json").read_text(encoding="utf-8")); s["pstar"] = "S0|none" if s["pstar"] != "S0|none" else "SZ|bal"
    (md / "specs.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    mut.append(dict(变异="P* 改名", 检出=bool(equiv_compare(md, NR.RUNS / NR.tag_of(*args[0][1:])))))
    shutil.rmtree(md)
    summ = dict(全部一致=all(r["一致"] for r in rep), 单元=rep, 变异对照=mut, 变异全检出=all(m["检出"] for m in mut),
                adapter_version=ADAPTER_VERSION, code_sha=code_sha(), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "R45_TARGET_EQUIV.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


def _equiv_entry(a):
    try:
        return run_unit(*a, runs=OUT / "equiv_runs")
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e} {traceback.format_exc()[-800:]}"


if __name__ == "__main__":
    if sys.argv[1] == "one":
        print(run_unit(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])))
    elif sys.argv[1] == "run":
        print(run(sys.argv[2].split(","), int(sys.argv[3])))
    elif sys.argv[1] == "equiv":
        equiv()
