# -*- coding: utf-8 -*-
r"""R43 T1 端到端检查（运行 320 个重复之前）。每项都配一个必须失败的变异对照。

E1 真值隔离：把评价集 p* 随机打乱后，所有选择器的决定与依据量逐位不变；
   变异对照 = 故意用 p* 挑模型的「作弊」选择器，打乱后依据量必须改变。
E2 确定性：同一重复跑两次，各臂真值风险与选择器结果逐位一致；变异对照 = 换重复号必须不同。
E3 生成器：P2 的 A 轨只依赖 X0–X3、B 轨只依赖 X4–X7；N1 与 Z 无关；实际 π_A 与设定一致；无缺失；分组 = 个体；
   阈值变体 Z = 1[X8 > Φ⁻¹(1−π)]。变异对照：扰动 A 轨的 X0 必须改变 p*；N2 必须依赖 Z。
E4 评价器：条件 Brier 与按 p* 抽 Y 的经验 Brier 在 4 个标准误内一致；变异对照 = 漏掉 p*(1−p*) 项必须偏离。
E5 认证后不重拟合：FCT/USTUN 部署风险与对应 F 拟合臂的风险逐位相同。
端到端检查用缩小的网格（PGSS_T1_SMOKE=1，n=2000），只检查程序行为，不产生任何结果数字。
"""
import sys, os, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PGSS_T1_SMOKE"] = "1"
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import norm
import pgss_r43_t1 as T1
import pgss_xgb_sim0 as S0

KEYS = ["选择器", "轨", "专业化", "部署", "t", "依据1", "依据2"]


def same(a, b, cols):
    a = a[cols].reset_index(drop=True); b = b[cols].reset_index(drop=True)
    return bool(a.equals(b))


def main():
    t0 = time.time(); R = {}
    args = ("P2", 2000, 0.3, 0)
    m1, A1, S1 = T1.run_rep(*args, write=False)
    m2, A2, S2 = T1.run_rep(*args, write=False, shuffle_pe=True)
    c1 = T1.run_rep(*args, write=False, cheat=True)[2]
    c2 = T1.run_rep(*args, write=False, cheat=True, shuffle_pe=True)[2]
    R["E1_打乱p星后选择器不变"] = same(S1, S2, KEYS)
    R["E1_变异_作弊选择器被检出"] = not same(c1, c2, KEYS)
    m3, A3, S3 = T1.run_rep(*args, write=False)
    R["E2_同一重复逐位一致"] = bool(A1.equals(A3) and S1.equals(S3))
    A4 = T1.run_rep("P2", 2000, 0.3, 1, write=False)[1]
    R["E2_变异_换重复号不同"] = not A1.equals(A4)
    # E3 生成器
    rng = np.random.default_rng(5); X = S0.gen_X(20000, rng); z = rng.random(20000) < 0.3
    p = S0.pstar(X, z, "P2")
    Xa = X.copy(); Xa[:, 4:8] += 1.0
    Xb = X.copy(); Xb[:, 0:4] += 1.0
    R["E3_P2_A轨不依赖X4到X7"] = bool(np.array_equal(S0.pstar(Xa, z, "P2")[z], p[z]))
    R["E3_P2_B轨不依赖X0到X3"] = bool(np.array_equal(S0.pstar(Xb, z, "P2")[~z], p[~z]))
    Xc = X.copy(); Xc[:, 0] += 1.0
    R["E3_变异_扰动A轨X0必须改变p星"] = not np.array_equal(S0.pstar(Xc, z, "P2")[z], p[z])
    R["E3_N1与Z无关"] = bool(np.array_equal(S0.pstar(X, z, "N1"), S0.pstar(X, ~z, "N1")))
    R["E3_变异_N2依赖Z"] = not np.array_equal(S0.pstar(X, z, "N2"), S0.pstar(X, ~z, "N2"))
    D, Ev, pe, _, pdev = T1.make_data("P1", 12000, 0.15, 3)
    se = np.sqrt(0.15 * 0.85 / T1.N_EVAL)
    R["E3_评价集实际πA在4SE内"] = bool(abs(Ev["z"].mean() - 0.15) < 4 * se)
    R["E3_开发集无缺失"] = bool(not np.isnan(D["X"]).any())
    R["E3_分组为个体"] = bool(len(np.unique(D["g"])) == len(D["g"]))
    Dt, Evt, pet, _, _ = T1.make_data("P2-T", 6000, 0.30, 0)
    R["E3_阈值变体Z等于X8阈值"] = bool(np.array_equal(Evt["z"], Evt["X"][:, T1.THR_COL] > norm.ppf(0.70)))
    R["E3_变异_独立Z不等于阈值"] = not np.array_equal(Ev["z"], Ev["X"][:, T1.THR_COL] > norm.ppf(0.85))
    # E4 评价器
    rr = np.random.default_rng(9); ph = np.clip(pe + rr.normal(0, 0.05, len(pe)), 0.001, 0.999)
    Y = (rr.random((20, len(pe))) < pe).astype(float)
    emp = ((Y - ph) ** 2).mean(axis=1); cond = T1.unit_loss(ph, pe).mean()
    sd = emp.std(ddof=1) / np.sqrt(len(emp))
    R["E4_条件Brier与经验Brier一致"] = bool(abs(emp.mean() - cond) < 4 * sd)
    bad = ((pe - ph) ** 2).mean()
    R["E4_变异_漏掉p星方差项被检出"] = bool(abs(emp.mean() - bad) > 4 * sd)
    # E5 认证后不重拟合
    ar = A1.set_index("臂")
    ok = True
    for nm in ("FCT|P*", "FCT|S0*", "USTUN_ADAPT"):
        for g in ("A", "B"):
            r = S1[(S1.选择器 == nm) & (S1.轨 == g)].iloc[0]
            ok &= bool(r.R == ar.loc[r.部署, f"R_{g}"])
    R["E5_认证部署即F拟合臂"] = ok
    out = dict(全部通过=all(R.values()), 结果=R, 秒=round(time.time() - t0))
    T1.T1DIR.mkdir(parents=True, exist_ok=True)
    (T1.T1DIR / "R43_T1_E2E_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
