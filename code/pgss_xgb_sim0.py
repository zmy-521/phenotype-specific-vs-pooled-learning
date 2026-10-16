# -*- coding: utf-8 -*-
r"""Stage 0B 小规模模拟：只验证实现、检查明显失控、展示预定机制能被表达（不是 T1/T2，不估计错误率）。

四个场景（GPT 第 46 轮 §9 0B）
  N1 无关系差异       ：两轨同一条件风险函数，Z 与 X 独立
  N2 只有截距差异     ：logit 加 0.8·Z
  P1 有差异但 SZ 可学 ：Z 与单个特征的线性交互（树模型的合并臂可在 Z 上分裂学到）
  P2 轨道特异稀疏结构 ：A 轨只依赖 X0–X3（非线性+交互），B 轨只依赖 X4–X7

真值：给定生成概率 p*，测试集条件 Brier = p*(1−p*) + (p*−p̂)²，在 4 万例无标签 X 上解析计算；
      选择器只看到可观察的 Y（开发集），从不接触 p*。
程序：与正式实验同一引擎与同一网格（完整 develop + final_predict）。开发集 n=6000（A 轨约 30%）。
输出：22_R41_Stage0/sim0/R41_SIM0.parquet 与汇总
"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

OUT = HERE.parent / "22_R41_Stage0" / "sim0"
SCEN = ["N1", "N2", "P1", "P2"]
REPS = 6
N_DEV, N_EVAL, PI_A, P = 6000, 40000, 0.3, 23


def gen_X(n, rng):
    C = 0.2 * np.ones((P, P)) + 0.8 * np.eye(P)
    return rng.multivariate_normal(np.zeros(P), C, size=n)


def pstar(X, z, scen):
    common = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.4 * np.tanh(X[:, 2] * X[:, 3]) + 0.3 * X[:, 5]
    if scen == "N1":
        lin = -1.6 + common
    elif scen == "N2":
        lin = -1.9 + common + 0.8 * z
    elif scen == "P1":
        lin = -1.6 + common + 1.0 * z * X[:, 6]
    elif scen == "P2":
        gA = 1.2 * np.tanh(1.5 * X[:, 0]) + 0.9 * X[:, 1] * X[:, 2] - 0.6 * (X[:, 3] ** 2 - 1)
        gB = 1.0 * X[:, 4] - 0.8 * np.abs(X[:, 5]) + 0.7 * X[:, 6] * X[:, 7]
        lin = np.where(z, -1.4 + gA, -1.7 + gB)
    else:
        raise ValueError(scen)
    return 1 / (1 + np.exp(-lin))


def one(args):
    scen, rep = args
    import os, pgss_xgb_engine as EN
    if os.environ.get("PGSS_SIM0_SMOKE") == "1":
        EN.GRID_TRACK = EN.GRID_TRACK[:1]; EN.GRID_POOLED = EN.GRID_POOLED[:1]; EN.K_GRID = [5, 23]; EN.MAX_ROUNDS = 60
    rng = np.random.default_rng(1000 * SCEN.index(scen) + rep)
    Xd = gen_X(N_DEV, rng); zd = rng.random(N_DEV) < PI_A
    pd_ = pstar(Xd, zd, scen); yd = (rng.random(N_DEV) < pd_).astype(int)
    Xd[rng.random(Xd.shape) < 0.05] = np.nan
    Xe = gen_X(N_EVAL, rng); ze = rng.random(N_EVAL) < PI_A; pe = pstar(Xe, ze, scen)
    D = dict(X=Xd, y=yd, z=zd, g=(np.arange(N_DEV) // 3).astype(str), uid=np.arange(N_DEV, dtype=np.int64))
    Te = dict(X=Xe, y=np.zeros(N_EVAL, int), z=ze, g=np.arange(N_EVAL).astype(str),
              uid=np.arange(N_EVAL, dtype=np.int64) + 10**7)
    lay = EN.task_layout(Xd, zd, [f"x{i}" for i in range(P)])
    EN.reset_caches(); t0 = time.time()
    r = EN.develop(D, lay, 7 + rep)
    fp = EN.final_predict(r["specs"], D, Te, lay)
    rows = []
    for arm, v in fp.items():
        ph = v[v["chosen"]]
        loss = pe * (1 - pe) + (pe - ph) ** 2
        rows.append(dict(场景=scen, rep=rep, 臂=arm, 是P星=(arm == r["pstar"]), 校准=v["chosen"],
                         R_A=float(loss[ze].mean()), R_B=float(loss[~ze].mean())))
    loss_or = pe * (1 - pe)
    rows.append(dict(场景=scen, rep=rep, 臂="ORACLE(p*)", 是P星=False, 校准="-",
                     R_A=float(loss_or[ze].mean()), R_B=float(loss_or[~ze].mean())))
    return rows, dict(场景=scen, rep=rep, pstar=r["pstar"], sec=round(time.time() - t0, 1), fits=len(EN.FITLOG),
                      kA=r["specs"]["HTFS"]["kA"], kB=r["specs"]["HTFS"]["kB"])


def main(jobs=8, smoke=False):
    from concurrent.futures import ProcessPoolExecutor
    global OUT
    if smoke:
        OUT = OUT.parent / "smoke_sim0"
    OUT.mkdir(parents=True, exist_ok=True)
    jl = [(s, r) for s in SCEN for r in range(1 if smoke else REPS)]
    rows, meta = [], []
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for rr, mm in ex.map(one, jl):
            rows += rr; meta.append(mm); print(mm, flush=True)
    R = pd.DataFrame(rows); Mt = pd.DataFrame(meta)
    R["R_bal"] = (R.R_A + R.R_B) / 2
    # 每个复制内相对 P* 的真值相对改善（正值 = 该臂真实更好）
    ref = R[R.是P星].set_index(["场景", "rep"])[["R_A", "R_B", "R_bal"]]
    R = R.join(ref, on=["场景", "rep"], rsuffix="_Pstar")
    for c in ("A", "B", "bal"):
        R[f"真值相对改善_{c}"] = (R[f"R_{c}_Pstar"] - R[f"R_{c}"]) / R[f"R_{c}_Pstar"]
    R.to_parquet(OUT / "R41_SIM0.parquet", index=False); Mt.to_parquet(OUT / "R41_SIM0_meta.parquet", index=False)
    (OUT / "_DONE").write_text("ok", encoding="utf-8")
    print("done")


if __name__ == "__main__":
    import os
    sm = "--smoke" in sys.argv
    if sm:
        os.environ["PGSS_SIM0_SMOKE"] = "1"
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8, smoke=sm)
