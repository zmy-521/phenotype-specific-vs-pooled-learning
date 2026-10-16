# -*- coding: utf-8 -*-
r"""R43 T1 扩展（N1-T、P2-T）的端到端小测试（GPT 第48轮补充裁决 §3）。每项配必须失败的变异对照。

X1 Z 的确定性：开发集与评价集都逐例满足 Z = 1[X8 > Φ⁻¹(1−πA)]；实际 A 轨比例在 4 个二项标准误内。
   变异：原 N1（Z 独立生成）必须不满足。
X2 X8 边际为标准正态：大样本均值、方差在 4 个标准误内，KS 检验 p > 0.001。变异：X8+0.1 必须被检出。
X3 结局函数只直接使用 X0–X7：固定 Z，扰动 X8–X22，p* 逐例不变（N1-T 与 P2-T）。变异：扰动 X0 必须改变 p*。
X4 N1-T 无条件表型效应：p*(X, Z) = p*(X, 1−Z)。变异：P2-T 翻转 Z 必须改变 p*。
X5 P2-T 经 Z 间接依赖 X8：把 X8 移过阈值（其余不变）必须改变部分 p*；N1-T 同样操作 p* 不变。
X6 样本划分：分组 = 个体；认证集约占 1/3；内层 5 折按组且不交。
X7 损失计算：ORACLE 臂的各轨风险 = 该轨 p*(1−p*) 的均值（Bayes 风险）；选择器在打乱评价 p* 后决定与依据量逐位不变；
   作弊选择器（用 p* 挑模型）在打乱后必须改变（变异对照）。
X8 输出位置：经扩展入口写出的重复只出现在临时 runs_ext，不触碰原 runs/。
端到端部分用缩小网格（PGSS_T1_SMOKE=1，n=2000）。
"""
import sys, os, json, pathlib, time, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PGSS_T1_SMOKE"] = "1"
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import norm, kstest
import pgss_r43_t1 as T1
import pgss_r43_t1_ext as TX
import pgss_xgb_sim0 as S0

KEYS = ["选择器", "轨", "专业化", "部署", "t", "依据1", "依据2"]


def main():
    t0 = time.time(); R = {}
    # X1
    ok = True
    for sc in TX.SCEN:
        for pi in T1.PI_GRID:
            D, Ev, pe, _, _ = T1.make_data(sc, 6000, pi, 0)
            c = norm.ppf(1 - pi)
            ok &= bool(np.array_equal(D["z"], D["X"][:, 8] > c) and np.array_equal(Ev["z"], Ev["X"][:, 8] > c))
            ok &= bool(abs(Ev["z"].mean() - pi) < 4 * np.sqrt(pi * (1 - pi) / T1.N_EVAL))
    R["X1_Z规则逐例成立且A轨比例正确"] = ok
    D0, Ev0, *_ = T1.make_data("N1", 6000, 0.15, 0)
    R["X1_变异_独立Z不满足规则"] = not np.array_equal(Ev0["z"], Ev0["X"][:, 8] > norm.ppf(0.85))
    # X2
    x8 = T1.make_data("N1-T", 6000, 0.15, 1)[1]["X"][:, 8]
    se = 1 / np.sqrt(len(x8))
    good = abs(x8.mean()) < 4 * se and abs(x8.var(ddof=1) - 1) < 4 * np.sqrt(2 / len(x8)) and kstest(x8, "norm").pvalue > 1e-3
    bad = x8 + 0.1
    R["X2_X8边际为标准正态"] = bool(good)
    R["X2_变异_平移0.1被检出"] = bool(not (abs(bad.mean()) < 4 * se and kstest(bad, "norm").pvalue > 1e-3))
    # X3、X4、X5
    rng = np.random.default_rng(3); X = S0.gen_X(30000, rng); z = X[:, 8] > norm.ppf(0.7)
    ok3 = True
    for base in ("N1", "P2"):
        p = S0.pstar(X, z, base); Xp = X.copy(); Xp[:, 8:] += rng.normal(0, 1, Xp[:, 8:].shape)
        ok3 &= bool(np.array_equal(S0.pstar(Xp, z, base), p))
    R["X3_固定Z时只依赖X0到X7"] = ok3
    Xq = X.copy(); Xq[:, 0] += 1.0
    R["X3_变异_扰动X0改变p星"] = not np.array_equal(S0.pstar(Xq, z, "P2"), S0.pstar(X, z, "P2"))
    R["X4_N1T无条件表型效应"] = bool(np.array_equal(S0.pstar(X, z, "N1"), S0.pstar(X, ~z, "N1")))
    R["X4_变异_P2T翻转Z改变p星"] = not np.array_equal(S0.pstar(X, z, "P2"), S0.pstar(X, ~z, "P2"))
    Xs = X.copy(); Xs[:, 8] = np.where(z, -5.0, 5.0); zs = Xs[:, 8] > norm.ppf(0.7)
    R["X5_P2T经Z间接依赖X8"] = bool(not np.array_equal(S0.pstar(Xs, zs, "P2"), S0.pstar(X, z, "P2")))
    R["X5_N1T移动X8不改变p星"] = bool(np.array_equal(S0.pstar(Xs, zs, "N1"), S0.pstar(X, z, "N1")))
    # X6
    import pgss_xgb_engine as EN
    D, Ev, pe, _, _ = T1.make_data("P2-T", 6000, 0.3, 0)
    keep, hold = EN.cert_split(D["g"], 7 + 50_000)
    folds = EN.group_split(D["g"], EN.INNER_FOLDS, 7)
    R["X6_分组为个体"] = bool(len(np.unique(D["g"])) == len(D["g"]))
    R["X6_认证集约三分之一"] = bool(abs(len(hold) / len(D["g"]) - 1 / 3) < 0.01)
    R["X6_内层五折不交且覆盖"] = bool(all(np.intersect1d(tr, te).size == 0 for tr, te in folds)
                                  and np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(len(D["g"]))))
    # X7、X8（端到端，缩小网格）
    tmp = pathlib.Path(tempfile.mkdtemp()); orig = T1.RUNS
    TX.RUNS_EXT = tmp / "runs_ext"
    m1 = TX.run_one("P2-T", 2000, 0.3, 0)
    od = TX.RUNS_EXT / T1.tag_of("P2-T", 2000, 0.3, 0)
    A1 = pd.read_parquet(od / "arms.parquet").set_index("臂"); S1 = pd.read_parquet(od / "sel.parquet")
    T1.RUNS = orig
    D, Ev, pe, _, _ = T1.make_data("P2-T", 2000, 0.3, 0)
    z = Ev["z"]
    R["X7_ORACLE风险等于Bayes风险"] = bool(np.isclose(A1.loc["ORACLE", "R_A"], (pe[z] * (1 - pe[z])).mean(), rtol=0, atol=1e-15)
                                     and np.isclose(A1.loc["ORACLE", "R_B"], (pe[~z] * (1 - pe[~z])).mean(), rtol=0, atol=1e-15))
    S2 = T1.run_rep("P2-T", 2000, 0.3, 0, write=False, shuffle_pe=True)[2]
    R["X7_打乱p星后选择器不变"] = bool(S1[KEYS].reset_index(drop=True).equals(S2[KEYS].reset_index(drop=True)))
    c1 = T1.run_rep("N1-T", 2000, 0.3, 0, write=False, cheat=True)[2]
    c2 = T1.run_rep("N1-T", 2000, 0.3, 0, write=False, cheat=True, shuffle_pe=True)[2]
    R["X7_变异_作弊选择器被检出"] = not c1[KEYS].reset_index(drop=True).equals(c2[KEYS].reset_index(drop=True))
    ex = json.loads((od / "ext.json").read_text(encoding="utf-8"))
    R["X8_扩展输出只写入runs_ext"] = bool(od.exists() and not (orig / od.name).exists() and ex["开发_Z规则逐例一致"]
                                     and ex["评价_Z规则逐例一致"])
    out = dict(全部通过=all(R.values()), 结果=R, 秒=round(time.time() - t0))
    (T1.T1DIR / "R43_T1_EXT_E2E_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
