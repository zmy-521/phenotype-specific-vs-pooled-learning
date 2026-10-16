# -*- coding: utf-8 -*-
r"""PGSS XGB 正式运行单元（引擎 v1.0，SAP v1.0）。一个单元 = 任务 × seed × 外层折 × 分支（main | cert | lc[比例]）。

main ：develop(D) → 全部 11 臂在 D 上冻结拟合 → 外层测试 raw/cal 预测（P*_D 即其中被选者）
cert ：D 按组划 F/C → develop(F；合并全部 + Hfull + HTFS) → P*_F 与 HTFS_F 预测 C 与外层测试
       → 逐轨认证（MIMIC 正态参照；eICU t 参照，自由度=认证集分量数−1；每轨 α=0.025）→ 决定
       → 认证后程序的测试预测（专业化的轨用 HTFS_F，其余用 P*_F）；**认证后不用 C 的标签重拟合任何模型**
lc   ：开发集按组嵌套子集（比例 25/50/75%）→ develop(合并全部 + Hfull + HTFS) → P*/Hfull/HTFS 测试预测

产物（目录 runs/<tag>/）：preds_test.parquet（含 unit_id，**只留本地，不打包**）、preds_C.parquet（cert）、
specs.json、inner_scores.parquet、fitlog.parquet（面板/轮数/叶节点/耗时）、callog.parquet、cert.json、meta.json、_DONE。
失败：写 _FAILED（任务/seed/折/分支/角色/信息），不搜索替代种子、不删候选、不记为零收益。
"""
import sys, os, json, pathlib, time, argparse, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

RUNS = HERE.parent / "23_R42_正式主实验" / "runs"
M = 0.01
ALPHA_TRACK = 0.025
CODE_FILES = ["pgss_xgb_engine.py", "pgss_xgb_run.py", "pgss_r36_engine.py", "pgss_r38_bench.py"]


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def tag_of(db, oc, ph, seed, fold, branch, frac=None):
    return f"{db}__{oc}__{ph}__{seed}__{fold}__{branch}" + (f"__{int(round(100 * frac))}" if frac else "")


def fsha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def run_unit(db, oc, ph, seed, fold, branch, frac=None, data_fp=None):
    import psutil
    import pgss_r38_bench as BN, pgss_r36_engine as E, pgss_xgb_engine as EN
    from pgss_xgb_pathaudit import task_arrays
    tag = tag_of(db, oc, ph, seed, fold, branch, frac)
    od = RUNS / tag; od.mkdir(parents=True, exist_ok=True)
    csha = code_sha()
    if (od / "_DONE").exists():
        m = json.loads((od / "meta.json").read_text(encoding="utf-8"))
        if m.get("engine_version") == EN.ENGINE_VERSION and m.get("code_sha") == csha:
            return "SKIP_DONE"
    (od / "_FAILED").unlink(missing_ok=True)
    t0 = time.time(); proc = psutil.Process()
    A, names, gs = task_arrays(db, oc, ph)
    folds = E.group_split(A["X"], A["y"], A["z"], gs, BN.K_OUTER, seed)      # = LR 冻结外层划分
    dev, te = folds[fold]
    D, Te = EN.sub(A, dev), EN.sub(A, te)
    lay = EN.task_layout(D["X"], D["z"], names)
    EN.set_namespace(db=db, oc=oc, ph=ph, seed=seed, fold=fold, branch=branch, frac=frac, data_fp=data_fp, code=csha)
    meta = dict(tag=tag, db=db, oc=oc, ph=ph, seed=seed, fold=fold, branch=branch, frac=frac,
                engine_version=EN.ENGINE_VERSION, code_sha=csha, data_fp=data_fp, z_name=lay["z_name"],
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                xgb_nthread=EN.PARAM_BASE["nthread"], n_dev=len(dev), n_test=len(te))
    out = pd.DataFrame({"unit_id": Te["uid"], "y": Te["y"], "z": Te["z"]})
    try:
        if branch == "main":
            r = EN.develop(D, lay, seed * 100 + fold)
            fp = EN.final_predict(r["specs"], D, Te, lay)
            for k, v in fp.items():
                out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
            meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fp.items()})
        elif branch == "cert":
            keep, hold = EN.cert_split(D["g"], seed * 100 + fold + 50_000)
            F, C = EN.sub(D, keep), EN.sub(D, hold)
            r = EN.develop(F, lay, seed * 100 + fold + 70_000, arms_tracked=("Hfull", "HTFS"))
            ref, spe = r["specs"][r["pstar"]], r["specs"]["HTFS"]
            pc_ref, pc_s = EN.program_predict(ref, F, C, lay), EN.program_predict(spe, F, C, lay)
            pt_ref, pt_s = EN.program_predict(ref, F, Te, lay), EN.program_predict(spe, F, Te, lay)
            refd = "normal" if db == "mimic" else "t"
            cert = {}
            for g in ("A", "B"):
                res = EN.cert_test(C["y"], pc_ref, pc_s, C["z"], C["g"], M, g, ref=refd)
                res_n = EN.cert_test(C["y"], pc_ref, pc_s, C["z"], C["g"], M, g, ref="normal")
                cert[g] = dict(primary=res, normal=res_n,
                               specialize=bool(np.isfinite(res["p"]) and res["p"] < ALPHA_TRACK))
            prog = np.where(Te["z"], pt_s if cert["A"]["specialize"] else pt_ref, pt_s if cert["B"]["specialize"] else pt_ref)
            out["Pstar_F"] = pt_ref; out["HTFS_F"] = pt_s; out["certified_program"] = prog
            pd.DataFrame({"unit_id": C["uid"], "y": C["y"], "z": C["z"], "g": C["g"], "Pstar_F": pc_ref,
                          "HTFS_F": pc_s}).to_parquet(od / "preds_C.parquet", index=False)
            (od / "cert.json").write_text(json.dumps(cert, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
            meta.update(pstar_F=r["pstar"], n_F=len(keep), n_C=len(hold), cert_reference=refd,
                        decision=f"{'S' if cert['A']['specialize'] else 'P'}{'S' if cert['B']['specialize'] else 'P'}")
        elif branch == "lc":
            si = EN.nested_fraction(D["g"], frac, seed * 10 + fold)
            S = EN.sub(D, si)
            r = EN.develop(S, lay, seed * 100 + fold, arms_tracked=("Hfull", "HTFS"))
            fp = EN.final_predict(r["specs"], S, Te, lay, names=[r["pstar"], "Hfull", "HTFS"])
            for k, v in fp.items():
                out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
            meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fp.items()}, n_sub=len(si),
                        n_sub_groups=int(len(np.unique(S["g"]))))
        else:
            raise ValueError(branch)
    except EN.FoldFailure as ex:
        last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
        (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last,
                                                    trace=traceback.format_exc()[-1500:]), ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        return "FAILED"
    out.to_parquet(od / "preds_test.parquet", index=False)
    (od / "specs.json").write_text(json.dumps(dict(specs=r["specs"], inner=r["inner"], pstar=r["pstar"],
                                                   inner_folds=r["inner_folds"]), ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    pd.DataFrame(r["rows"]).to_parquet(od / "inner_scores.parquet", index=False)
    FL = pd.DataFrame(EN.FITLOG); FL.to_parquet(od / "fitlog.parquet", index=False)
    pd.DataFrame(EN.CALLOG).to_parquet(od / "callog.parquet", index=False)
    meta.update(sec_wall=round(time.time() - t0, 1), n_fits=len(FL),
                sec_fit_elapsed=round(float((FL.sec_es + FL.sec_refit).sum()), 1),
                peak_rss_mb=round(proc.memory_info().peak_wset / 2**20, 1) if hasattr(proc.memory_info(), "peak_wset")
                else round(proc.memory_info().rss / 2**20, 1),
                requests=dict(EN.REQLOG))
    meta["artifacts"] = {f.name: fsha(f) for f in sorted(od.glob("*.parquet"))}
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (od / "_DONE").write_text(str(meta["sec_wall"]), encoding="utf-8")
    return "OK"


def _unit_entry(args):
    try:
        return args, run_unit(*args)
    except Exception as e:                       # 非预期异常：记录，不中断整批
        tag = tag_of(*args[:7])
        od = RUNS / tag; od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"非预期异常 {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", encoding="utf-8")
        return args, "ERROR"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--db", "--oc", "--ph", "--branch"):
        ap.add_argument(a)
    ap.add_argument("--seed", type=int); ap.add_argument("--fold", type=int)
    ap.add_argument("--frac", type=float, default=None); ap.add_argument("--data_fp", default=None)
    a = ap.parse_args()
    print(run_unit(a.db, a.oc, a.ph, a.seed, a.fold, a.branch, a.frac, a.data_fp))
