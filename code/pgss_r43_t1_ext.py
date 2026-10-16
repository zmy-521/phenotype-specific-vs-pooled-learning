# -*- coding: utf-8 -*-
r"""R43 T1 扩展（GPT 第 48 轮补充裁决）：N1-T、P2-T × n{6000,12000} × πA{0.15,0.30} × 20 = 160 个重复。

**探索性扩展，不是 T2 确认**；与原 320 个重复分开标识、分开输出、分开汇总。
  - 输出 t1/runs_ext/<tag>/（原 320 在 t1/runs/，本文件不读写它们）；配置 t1/R43_T1_EXT_CONFIG.json（运行前写定并带指纹）。
  - 复用 pgss_r43_t1 的生成、学习程序、选择器与评价器（只 import，不修改该文件——原 320 运行期间文件指纹不变）。
  - 机制：Z = 1[X8 > Φ⁻¹(1−πA)]，开发集与评价集各自独立按此规则生成；X 的生成结构不变（X8 边际为标准正态）；
    直接进入结局函数的预测变量仍为 X0–X7（N1 的公式不用 Z；P2 的轨内结构不变，最终 p* 经 Z 间接依赖 X8）。
  - N1-T：条件预测关系无额外表型效应；因 X8 与其他预测变量相关，两轨的边际事件率可以不同（病例构成差异，不是条件关系差异）。
  - 评价集 100,000（与原 320 一致，≥ 40,000）；无额外缺失；个体级分组；选择器不读 p*。
  - 每个重复另记（ext.json）：各轨实际样本数、实际 A 轨比例、观察与期望事件率、Bayes 风险、Z 规则逐例核对、X8 边际。
用法：python pgss_r43_t1_ext.py config | run <jobs> [--only cost|all] | one <scen> <n> <pi> <rep>
"""
import sys, os, json, pathlib, time, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import norm
import pgss_r43_t1 as T1

RUNS_EXT = T1.T1DIR / "runs_ext"
SCEN = list(T1.SCEN_THR)                    # ["N1-T", "P2-T"]
EXT_FILES = ["pgss_r43_t1_ext.py"] + T1.CODE_FILES


def ext_sha():
    h = hashlib.sha256()
    for f in EXT_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def extra_stats(scen, n, pi, rep):
    """确定性重新生成同一重复的数据（与 run_rep 内部逐位相同），只算描述量；不接触任何模型。"""
    D, Ev, pe, _, pdev = T1.make_data(scen, n, pi, rep)
    c = norm.ppf(1 - pi); out = dict(scen=scen, n=n, pi=pi, rep=rep, 阈值=float(c))
    for lab, X, z, p, y in (("开发", D["X"], D["z"], pdev, D["y"]), ("评价", Ev["X"], Ev["z"], pe, None)):
        x8 = X[:, T1.THR_COL]
        out[f"{lab}_Z规则逐例一致"] = bool(np.array_equal(z, x8 > c))
        out[f"{lab}_A比例"] = float(z.mean()); out[f"{lab}_nA"] = int(z.sum()); out[f"{lab}_nB"] = int((~z).sum())
        out[f"{lab}_X8均值"] = float(x8.mean()); out[f"{lab}_X8标准差"] = float(x8.std(ddof=1))
        for g, m in (("A", z), ("B", ~z)):
            out[f"{lab}_期望事件率_{g}"] = float(p[m].mean())
            out[f"{lab}_Bayes风险_{g}"] = float((p[m] * (1 - p[m])).mean())
            if y is not None:
                out[f"{lab}_观察事件率_{g}"] = float(y[m].mean()); out[f"{lab}_事件_{g}"] = int(y[m].sum())
    return out


def run_one(scen, n, pi, rep):
    T1.RUNS = RUNS_EXT                     # 只改本进程内的输出目录
    ex = extra_stats(scen, n, pi, rep)
    meta, A, S = T1.run_rep(scen, n, pi, rep, write=True)
    od = RUNS_EXT / T1.tag_of(scen, n, pi, rep)
    (od / "ext.json").write_text(json.dumps(dict(ex, ext_sha=ext_sha()), ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def _entry(args):
    od = RUNS_EXT / T1.tag_of(*args)
    try:
        run_one(*args)
        return args, "OK"
    except Exception as e:
        od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", encoding="utf-8")
        return args, "FAILED"


def job_list(only):
    if only == "cost":
        return [(s, n, pi, 0) for s in SCEN for n in T1.N_GRID for pi in T1.PI_GRID]
    return [(s, n, pi, rep) for rep in range(T1.REPS) for n in sorted(T1.N_GRID, reverse=True) for pi in T1.PI_GRID for s in SCEN]


def done(a):
    od = RUNS_EXT / T1.tag_of(*a)
    return (od / "_DONE").exists() and (od / "ext.json").exists()


def run(jobs, only):
    from concurrent.futures import ProcessPoolExecutor
    RUNS_EXT.mkdir(parents=True, exist_ok=True)
    st = T1.T1DIR / f"_STATUS_ext_{only}.json"
    jl = [a for a in job_list(only) if not done(a)]
    tot = len(job_list(only)); ok = fail = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res == "FAILED"
            st.write_text(json.dumps(dict(批次=f"ext_{only}", 总数=tot, 先前已完成=tot - len(jl), 本次完成=ok, 失败=fail,
                                          已用分钟=round((time.time() - t0) / 60, 1), 更新=time.strftime("%m-%d %H:%M:%S")),
                                     ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    (T1.T1DIR / (f"_DONE_ext_{only}" if fail == 0 else f"_FAILED_ext_{only}")).write_text(str(fail), encoding="utf-8")


def write_config():
    import pgss_xgb_sim0 as S0, inspect
    base = json.loads((T1.T1DIR / "R43_T1_CONFIG.json").read_text(encoding="utf-8"))
    cfg = dict(版本="R43-T1-EXT v1", 性质="T1 探索性扩展（GPT 第48轮补充裁决批准），不是 T2 确认", 扩展代码指纹=ext_sha(),
               扩展文件SHA={f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16] for f in EXT_FILES},
               原T1配置指纹=base["配置指纹"], 原T1代码指纹=base["代码指纹"], 机制=SCEN,
               Z规则="Z = 1[X8 > Φ⁻¹(1−πA)]，开发集与评价集各自独立按此规则生成", 阈值列=T1.THR_COL,
               结局函数直接使用的预测变量="X0–X7（与原 N1、P2 相同的 pstar 公式）", 输出目录="t1/runs_ext/",
               n=T1.N_GRID, pi_A=T1.PI_GRID, 重复=T1.REPS, 评价样本=T1.N_EVAL, 追加评价样本=T1.N_EVAL_EXTRA, 边界m=T1.M,
               每轨alpha=T1.ALPHA, 缺失="无", 分组="独立个体", 选择器="与原 T1 相同（12 个）",
               种子规则="同原 T1：SeedSequence([20431, 机制序号, n, round(100π), rep])；机制序号 N1-T=4、P2-T=5",
               生成_gen_X=inspect.getsource(S0.gen_X), 生成_pstar=inspect.getsource(S0.pstar), 生成_gen=inspect.getsource(T1.gen))
    s = json.dumps(cfg, ensure_ascii=False, indent=1, default=str)
    cfg["配置指纹"] = hashlib.sha256(s.encode()).hexdigest()[:16]
    (T1.T1DIR / "R43_T1_EXT_CONFIG.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(cfg["配置指纹"], cfg["扩展代码指纹"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "config":
        write_config()
    elif cmd == "run":
        run(int(sys.argv[2]), sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "all")
    elif cmd == "one":
        print(json.dumps(extra_stats(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])), ensure_ascii=False, indent=1))
