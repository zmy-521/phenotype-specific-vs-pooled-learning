# -*- coding: utf-8 -*-
r"""R46 有限重算（方案 28_R46_最终核验与T2/outcome_fix/R46_OUTCOME_FIX_PROTOCOL_v1.0.md §1.4、§2、§3）。

三类单元（学习程序 = 冻结引擎 xgb-1.0 原样；适配器 r45-target-1.0 原样）：
  nwicu ：NWICU OUT_HOSP_MORT × 6 表型 × 3 seed × 5 折（住院死亡不可评价人次删除）
  nh44  ：NHANES R44 main，O_CVD10 × 受影响表型（死因未知者删除）
  nhpop ：NHANES R45 目标一致性 pop 追加对照，同一受影响任务；HTFS 冻结规格读 **v2 的 nh44 单元**
外层折保留：在原任务人群与原标签上调用 pgss_r36_engine.group_split（同 seed）得到原外层折，再在开发与测试部分内删除不可评价人次。
单元主体与原运行器逐行对应（pgss_r45_nwicu_run.run_unit、pgss_r44_nhanes_run.run_unit、pgss_r45_target.run_unit），只改数据来源、外层折与输出目录。
零修正模式（zero=True）：不删除任何人次，必须逐位复现原单元（方案 §3 等价测试）。
用法：python pgss_r46_rerun.py equiv | audit <jobs> | run <kinds,逗号分隔> <jobs> | foldcheck | one <kind> <oc> <ph> <seed> <fold>
"""
import sys, os, json, pathlib, time, hashlib, traceback, shutil
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
R46 = ROOT / "28_R46_最终核验与T2"
DIRS = {"nwicu": R46 / "nwicu_v2" / "runs", "nh44": R46 / "nhanes_v2" / "runs44", "nhpop": R46 / "nhanes_v2" / "runs_target"}
EQ = R46 / "outcome_fix" / "equiv_runs"
VERIFY = R46 / "outcome_fix" / "R46_OUTCOME_VERIFY.json"
CODE_FILES = ["pgss_r46_rerun.py", "pgss_r46_outcome_verify.py", "pgss_r45_nwicu_run.py", "pgss_r44_nhanes_run.py",
              "pgss_r45_target.py", "pgss_xgb_engine.py", "pgss_r36_engine.py"]


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def fsha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def mods(kind):
    import pgss_r45_nwicu_run as NW, pgss_r44_nhanes_run as NR
    return NW if kind == "nwicu" else NR


def tag_of(kind, oc, ph, seed, fold):
    import pgss_r45_target as TG
    return TG.tag_of("pop", oc, ph, seed, fold) if kind == "nhpop" else mods(kind).tag_of(oc, ph, seed, fold)


def orig_dir(kind, oc, ph, seed, fold):
    import pgss_r45_target as TG
    return (TG.RUNS if kind == "nhpop" else mods(kind).RUNS) / tag_of(kind, oc, ph, seed, fold)


def status(kind):
    """返回 (uid → 可评价, uid → v2 标签, 状态SHA)。"""
    if kind == "nwicu":
        S = pd.read_parquet(R46 / "nwicu_v2" / "_local_only" / "MORT_STATUS.parquet")
        return S.住院死亡可评价, S.OUT_HOSP_MORT_v2, json.loads(VERIFY.read_text(encoding="utf-8"))["NWICU"]["状态SHA"]
    S = pd.read_parquet(R46 / "nhanes_v2" / "_local_only" / "CVD_STATUS.parquet")
    return S.O_CVD10可评价, S.O_CVD10_v2, json.loads(VERIFY.read_text(encoding="utf-8"))["NHANES"]["状态SHA"]


def arrays(kind, oc, ph, zero=False):
    """原任务数组 A1（原标签，外层折用）、保留掩码 keep、修正标签的数组 Ac、列名、字符串分组、数据指纹。"""
    M = mods(kind)
    A1, names, gs = M.task_arrays(oc, ph)
    ev, yv2, ssha = status(kind)
    keep = np.ones(len(A1["uid"]), bool) if zero else ev.reindex(A1["uid"]).to_numpy(bool)
    y2 = A1["y"].copy()
    if not zero:
        lab = yv2.reindex(A1["uid"]).to_numpy(float)
        assert not np.isnan(lab[keep]).any(), "保留人次的 v2 标签缺失"
        y2[keep] = lab[keep].astype(int)
    Ac = dict(A1, y=y2)
    fp = dict(M.data_fp(), r46_status=("zero" if zero else ssha))
    return A1, keep, Ac, names, gs, fp


def preserved_split(A1, gs, keep, seed, fold, K):
    import pgss_r36_engine as E
    folds = E.group_split(A1["X"], A1["y"], A1["z"], gs, K, seed)
    dev, te = folds[fold]
    return dev[keep[dev]], te[keep[te]], dev, te


def run_unit(kind, oc, ph, seed, fold, runs=None, zero=False):
    import psutil
    import pgss_xgb_engine as EN
    M = mods(kind)
    runs = DIRS[kind] if runs is None else runs
    tag = tag_of(kind, oc, ph, seed, fold)
    od = runs / tag; od.mkdir(parents=True, exist_ok=True)
    csha = code_sha()
    if (od / "_DONE").exists():
        return "SKIP_DONE"
    (od / "_FAILED").unlink(missing_ok=True)
    t0 = time.time(); proc = psutil.Process()
    A1, keep, Ac, names, gs, fp = arrays(kind, oc, ph, zero)
    dev, te, dev0, te0 = preserved_split(A1, gs, keep, seed, fold, M.K_OUTER)
    D, Te = EN.sub(Ac, dev), EN.sub(Ac, te)
    lay = EN.task_layout(D["X"], D["z"], names)
    db = "nwicu" if kind == "nwicu" else "nhanes"
    meta = dict(tag=tag, kind=kind, db=db, oc=oc, ph=ph, seed=seed, fold=fold, engine_version=EN.ENGINE_VERSION, code_sha=csha,
                data_fp=fp, zero=zero, n_dev_orig=len(dev0), n_test_orig=len(te0), n_dev=len(dev), n_test=len(te),
                删除_dev=int(len(dev0) - len(dev)), 删除_test=int(len(te0) - len(te)), events_dev=int(D["y"].sum()),
                nA_dev=int(D["z"].sum()), eventsA_dev=int(D["y"][D["z"]].sum()), xgb_nthread=EN.PARAM_BASE["nthread"],
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    out = pd.DataFrame({"unit_id": Te["uid"], "y": Te["y"], "z": Te["z"]})
    if kind == "nhpop":
        import pgss_r45_target as TG
        src = (DIRS["nh44"] if not zero else mods("nh44").RUNS) / mods("nh44").tag_of(oc, ph, seed, fold)
        sp44 = json.loads((src / "specs.json").read_text(encoding="utf-8"))
        h = {k: v for k, v in sp44["specs"]["HTFS"].items() if k != "cal"}
        TG.activate("pop")
        EN.set_namespace(db=db, oc=oc, ph=ph, seed=seed, fold=fold, branch="target-pop", adapter=TG.ADAPTER_VERSION,
                         data_fp=fp, code=csha)
        meta.update(target="pop", adapter_version=TG.ADAPTER_VERSION, r44_specs_dir=str(src.relative_to(ROOT)),
                    r44_specs_sha=fsha(src / "specs.json"), htfs_frozen=h)
        try:
            r = TG.develop_pooled(D, lay, seed * 100 + fold, {"A": h["kA"], "B": h["kB"]})
            hs, hchosen = TG.htfs_switch(D, lay, r["folds"], h)
            fpred = EN.final_predict(r["specs"], D, Te, lay)
            for k, v in fpred.items():
                out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
            meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fpred.items()}, htfs_chosen_target=hchosen,
                        calls=dict(TG.CALLS))
        except EN.FoldFailure as ex:
            last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
            (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last, trace=traceback.format_exc()[-1500:]),
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            return "FAILED"
        finally:
            TG.deactivate()
        spec_doc = dict(target="pop", specs=r["specs"], inner=r["inner"], pstar=r["pstar"], inner_folds=r["inner_folds"],
                        htfs_frozen=h, htfs_inner=hs, htfs_chosen_target=hchosen)
    else:
        EN.set_namespace(db=db, oc=oc, ph=ph, seed=seed, fold=fold, branch="main", data_fp=fp, code=csha)
        try:
            r = EN.develop(D, lay, seed * 100 + fold)
            fpred = EN.final_predict(r["specs"], D, Te, lay)
            for k, v in fpred.items():
                out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
            meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fpred.items()})
        except EN.FoldFailure as ex:
            last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
            (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last, trace=traceback.format_exc()[-1500:]),
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            return "FAILED"
        spec_doc = dict(specs=r["specs"], inner=r["inner"], pstar=r["pstar"], inner_folds=r["inner_folds"])
    if EN.DRYRUN:
        (od / "_DONE").write_text("dryrun", encoding="utf-8")
        return "OK"
    out.to_parquet(od / "preds_test.parquet", index=False)
    (od / "specs.json").write_text(json.dumps(spec_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(r["rows"]).to_parquet(od / "inner_scores.parquet", index=False)
    FL = pd.DataFrame(EN.FITLOG); FL.to_parquet(od / "fitlog.parquet", index=False)
    pd.DataFrame(EN.CALLOG).to_parquet(od / "callog.parquet", index=False)
    meta.update(sec_wall=round(time.time() - t0, 1), n_fits=len(FL),
                peak_rss_mb=round(proc.memory_info().rss / 2**20, 1), requests=dict(EN.REQLOG))
    meta["artifacts"] = {f.name: fsha(f) for f in sorted(od.glob("*.parquet"))}
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (od / "_DONE").write_text(str(meta["sec_wall"]), encoding="utf-8")
    return "OK"


# ── 单元清单 ────────────────────────────────────────────────────
def tasks(kind):
    v = json.loads(VERIFY.read_text(encoding="utf-8"))
    if kind == "nwicu":
        import pgss_r45_nwicu_run as NW
        return [("OUT_HOSP_MORT", ph) for ph in NW.PHENOS] if v["NWICU"]["需要重跑"] else []
    return [tuple(t) for t in v["NHANES"]["重跑任务"]]


def units(kind):
    M = mods(kind)
    return [(kind, oc, ph, s, f) for (oc, ph) in tasks(kind) for s in M.SEEDS for f in range(M.K_OUTER)]


def _entry(args):
    try:
        return args, run_unit(*args)
    except Exception as e:
        od = DIRS[args[0]] / tag_of(*args); od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"非预期异常 {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", encoding="utf-8")
        return args, "ERROR"


def run(kinds, jobs):
    from concurrent.futures import ProcessPoolExecutor
    allu = [u for k in kinds for u in units(k)]
    jl = [a for a in allu if not (DIRS[a[0]] / tag_of(*a) / "_DONE").exists()]
    ok = fail = 0; t0 = time.time(); st = R46 / "outcome_fix" / f"_STATUS_{'+'.join(kinds)}.json"
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res in ("FAILED", "ERROR")
            st.write_text(json.dumps(dict(总数=len(allu), 本次待跑=len(jl), 完成=ok, 失败或不可执行=fail,
                                          已用分钟=round((time.time() - t0) / 60, 1), 更新=time.strftime("%m-%d %H:%M:%S")),
                                     ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    return ok, fail


# ── 干跑审计：真实引擎 DRYRUN 走完划分与检查，不训练 ───────────────
def _audit_entry(args):
    import pgss_xgb_engine as EN
    EN.DRYRUN = True
    ad = R46 / "outcome_fix" / "pathaudit_runs"
    try:
        res = run_unit(*args, runs=ad / args[0])
    except Exception as e:
        res = f"ERROR {type(e).__name__}: {e}"
    od = ad / args[0] / tag_of(*args)
    info = (od / "_FAILED").read_text(encoding="utf-8")[:300] if (od / "_FAILED").exists() else ""
    return dict(类=args[0], 结局=args[1], 表型=args[2], seed=args[3], 外层折=args[4], 结果=res, 信息=info)


def audit(jobs):
    from concurrent.futures import ProcessPoolExecutor
    allu = [u for k in ("nwicu", "nh44") for u in units(k)]
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        rows = list(ex.map(_audit_entry, allu))
    A = pd.DataFrame(rows)
    A.to_csv(R46 / "outcome_fix" / "R46_RERUN_PATHAUDIT.csv", index=False, encoding="utf-8-sig")
    s = dict(可执行=int((A.结果 == "OK").sum()), 总数=len(A), 不可执行=A[A.结果 != "OK"].to_dict("records"), code_sha=code_sha())
    (R46 / "outcome_fix" / "R46_RERUN_PATHAUDIT.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


# ── 等价测试（零修正必须逐位复现原单元）───────────────────────────
EQUIV = [("nwicu", "OUT_HOSP_MORT", "肥胖", 20261004, 1), ("nh44", "O_CVD10", "肥胖", 20261003, 2),
         ("nhpop", "O_CVD10", "肥胖", 20261003, 2)]


def compare(new_dir, old_dir):
    bad = []
    n = json.loads((new_dir / "specs.json").read_text(encoding="utf-8")); o = json.loads((old_dir / "specs.json").read_text(encoding="utf-8"))
    for key in ("specs", "inner", "pstar"):
        if n.get(key) != o.get(key):
            bad.append(f"{key} 不同")
    if "htfs_chosen_target" in o and n.get("htfs_chosen_target") != o["htfs_chosen_target"]:
        bad.append("HTFS 版本选择不同")
    Pn = pd.read_parquet(new_dir / "preds_test.parquet"); Po = pd.read_parquet(old_dir / "preds_test.parquet")
    if list(Pn.columns) != list(Po.columns) or not np.array_equal(Pn.unit_id.to_numpy(), Po.unit_id.to_numpy()):
        bad.append("列或测试单位不同")
    else:
        for c in Po.columns:
            if not np.array_equal(Pn[c].to_numpy(), Po[c].to_numpy()):
                bad.append(f"预测 {c} 不同")
    return bad


def _equiv_entry(a):
    try:
        return run_unit(*a, runs=EQ / a[0], zero=True)
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e} {traceback.format_exc()[-800:]}"


def equiv():
    from concurrent.futures import ProcessPoolExecutor
    if EQ.exists():
        shutil.rmtree(EQ)
    with ProcessPoolExecutor(max_workers=len(EQUIV), max_tasks_per_child=1) as ex:
        res = list(ex.map(_equiv_entry, EQUIV))
    rep = []
    for a, r in zip(EQUIV, res):
        nd = EQ / a[0] / tag_of(*a); od = orig_dir(*a)
        bad = compare(nd, od) if r == "OK" else [f"运行结果 {r}"]
        rep.append(dict(单元=f"{a[0]}:{tag_of(*a)}", 运行=r, 不一致项=bad, 一致=not bad))
    mut = []
    nd = EQ / EQUIV[0][0] / tag_of(*EQUIV[0]); md = EQ / "_mutation"
    shutil.copytree(nd, md)
    P = pd.read_parquet(md / "preds_test.parquet"); c = [x for x in P.columns if x.startswith("HTFS|")][0]
    P.loc[0, c] = P.loc[0, c] + 1e-12; P.to_parquet(md / "preds_test.parquet", index=False)
    mut.append(dict(变异="HTFS 测试预测 +1e-12", 检出=bool(compare(md, orig_dir(*EQUIV[0])))))
    s = json.loads((md / "specs.json").read_text(encoding="utf-8")); s["pstar"] = "S0|none" if s["pstar"] != "S0|none" else "SZ|bal"
    (md / "specs.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    mut.append(dict(变异="P* 改名", 检出=bool(compare(md, orig_dir(*EQUIV[0])))))
    shutil.rmtree(md)
    # 变异对照：非零修正（真实删除）下同一单元必须与原单元不同
    a = EQUIV[0]; dd = EQ / "_nonzero"
    r = run_unit(*a, runs=dd, zero=False)
    mut.append(dict(变异="真实删除（非零修正）运行同一单元", 检出=bool(r == "OK" and compare(dd / tag_of(*a), orig_dir(*a)))))
    shutil.rmtree(dd)
    summ = dict(全部一致=all(r["一致"] for r in rep), 单元=rep, 变异对照=mut, 变异全检出=all(m["检出"] for m in mut),
                code_sha=code_sha(), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (R46 / "outcome_fix" / "R46_RERUN_EQUIV.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


# ── 外层折归属核对（正式重跑之后）───────────────────────────────
def foldcheck():
    rows = []
    for kind in ("nwicu", "nh44", "nhpop"):
        ev = status(kind)[0]
        for u in units(kind):
            nd = DIRS[kind] / tag_of(*u); od = orig_dir(*u)
            if not (nd / "_DONE").exists():
                rows.append(dict(类=kind, 单元=tag_of(*u), 完成=False)); continue
            Pn = pd.read_parquet(nd / "preds_test.parquet", columns=["unit_id", "y"])
            Po = pd.read_parquet(od / "preds_test.parquet", columns=["unit_id", "y"])
            kept = Po[ev.reindex(Po.unit_id).to_numpy(bool)]
            same = np.array_equal(np.sort(Pn.unit_id.to_numpy()), np.sort(kept.unit_id.to_numpy()))
            ylab = same and np.array_equal(Pn.sort_values("unit_id").y.to_numpy(), kept.sort_values("unit_id").y.to_numpy())
            mn = json.loads((nd / "meta.json").read_text(encoding="utf-8"))
            rows.append(dict(类=kind, 单元=tag_of(*u), 完成=True, 原测试人次=len(Po), 删除人次=int(len(Po) - len(kept)),
                             v2测试人次=len(Pn), 测试集合等于原集合去掉排除者=bool(same), 保留者标签相同=bool(ylab),
                             开发删除=mn["删除_dev"], 测试删除=mn["删除_test"]))
    F = pd.DataFrame(rows)
    F.to_csv(R46 / "outcome_fix" / "R46_FOLDCHECK.csv", index=False, encoding="utf-8-sig")
    ok = bool(F.完成.all() and F.测试集合等于原集合去掉排除者.all() and F.保留者标签相同.all())
    print(F.groupby("类")[["完成", "测试集合等于原集合去掉排除者", "保留者标签相同"]].all().to_string(), "\n全部通过:", ok)
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "equiv":
        equiv()
    elif cmd == "audit":
        audit(int(sys.argv[2]))
    elif cmd == "run":
        print(run(sys.argv[2].split(","), int(sys.argv[3])))
    elif cmd == "foldcheck":
        sys.exit(0 if foldcheck() else 1)
    elif cmd == "one":
        print(run_unit(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])))
