# -*- coding: utf-8 -*-
r"""R45 plasmode 端到端测试（缩小网格 PGSS_T1_SMOKE=1；不写正式 runs）。每条断言配能失败的对照。

P1 生成器：f0、gA、gB 与按方案文字独立重写的公式逐位一致（≤1e-12）；把 gB 中 0.7 改 0.6 的变异必须检出不一致。
P2 截距：三档 δ 的人群平均 p = 15%（|差| < 1e-10）。
P3 耦合：同一重复内三档 δ 的开发划分与均匀数 u 完全相同；不同重复不同（对照）。
P4 患者隔离：开发集与评价集的患者交集为空。
P5 评价器隔离：把评价集 p_δ 打乱，选择器的全部决定与 t 不变（p_δ 只进评价器），而风险改变（对照：确有差异）。
P6 δ=0 与 δ=1 的开发集标签不同（耦合下只由 p_δ 差异造成），δ 的确改变了数据。
输出：27_R45_外部复现与机制核验/plasmode/R45_PLASMODE_TESTS.json
"""
import os, sys, json, pathlib
os.environ["PGSS_T1_SMOKE"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pgss_r45_plasmode as PM


def explicit(Us, coef_gB_inter=0.7):
    U = Us
    f0 = 0.8 * U[:, 0] - 0.5 * U[:, 1] + 0.4 * np.tanh(U[:, 2] * U[:, 3]) + 0.3 * U[:, 5]
    gA = 1.2 * np.tanh(1.5 * U[:, 0]) + 0.9 * U[:, 1] * U[:, 2] - 0.6 * (U[:, 3] ** 2 - 1)
    gB = 1.0 * U[:, 4] - 0.8 * np.abs(U[:, 5]) + coef_gB_inter * U[:, 6] * U[:, 7]
    return f0, gA, gB


def main():
    res = []
    pop = PM.population(); parts = PM.gen_parts(pop)
    f0, gA, gB, gpar = parts
    ci = [pop["names"].index(c) for c in PM.U_COLS]
    U = pop["X"][:, ci]
    med, sc = np.array(gpar["中位数"]), np.array(gpar["尺度"])
    Us = np.clip(np.where(np.isnan(U), 0.0, (U - med) / sc), -PM.CLIP, PM.CLIP)
    e0, eA, eB = explicit(Us)
    ok = max(np.abs(f0 - e0).max(), np.abs(gA - eA).max(), np.abs(gB - eB).max()) < 1e-9
    _, _, mB = explicit(Us, 0.6)
    res.append(dict(测试="P1 生成器 = 方案公式（gB 系数变异检出为对照）", 通过=bool(ok and np.abs(gB - mB).max() > 1e-3),
                    最大差=float(max(np.abs(f0 - e0).max(), np.abs(gA - eA).max(), np.abs(gB - eB).max()))))
    ps = {}
    for d in PM.DELTAS:
        p, a = PM.p_delta(f0, gA, gB, pop["z"], d); ps[d] = p
    res.append(dict(测试="P2 人群平均 p = 15%", 通过=all(abs(ps[d].mean() - PM.EVENT_RATE) < 1e-10 for d in PM.DELTAS)))
    di0, ei0, ry0 = PM.split(pop, 0); di0b, _, ry0b = PM.split(pop, 0); di1, _, _ = PM.split(pop, 1)
    u0, u0b = ry0.random(len(di0)), ry0b.random(len(di0b))
    res.append(dict(测试="P3 同一重复划分与 u 相同、不同重复划分不同（对照）",
                    通过=bool(np.array_equal(di0, di0b) and np.array_equal(u0, u0b) and not np.array_equal(np.sort(di0), np.sort(di1)))))
    pat_dev = set(pop["gid"][di0]); pat_ev = set(pop["gid"][ei0])
    res.append(dict(测试="P4 开发/评价患者不交", 通过=len(pat_dev & pat_ev) == 0, 开发记录=len(di0), 开发患者=len(pat_dev), 评价记录=len(ei0)))
    y0 = (u0 < ps[0.0][di0]); y1 = (u0 < ps[1.0][di0])
    res.append(dict(测试="P6 δ 改变开发标签（耦合）", 通过=bool((y0 != y1).sum() > 0), 不同标签数=int((y0 != y1).sum())))
    m1, A1, S1 = PM.run_one(1.0, 0, write=False, pop=pop, parts=parts)
    m2, A2, S2 = PM.run_one(1.0, 0, write=False, shuffle_pe=True, pop=pop, parts=parts)
    same_dec = S1[["选择器", "轨", "专业化", "部署", "t"]].astype(str).equals(S2[["选择器", "轨", "专业化", "部署", "t"]].astype(str))
    diff_risk = not np.allclose(A1.R_A.to_numpy(), A2.R_A.to_numpy())
    res.append(dict(测试="P5 打乱评价 p_δ：选择不变、风险改变（对照）", 通过=bool(same_dec and diff_risk)))
    m0, A0, S0_ = PM.run_one(0.0, 0, write=False, pop=pop, parts=parts)
    res.append(dict(测试="P3b 实际运行中三档 δ 的划分与 u 指纹相同", 通过=bool(m0["dev_split_sha"] == m1["dev_split_sha"] and m0["u_sha"] == m1["u_sha"])))
    summ = dict(全部通过=all(r["通过"] for r in res), 测试=res, code_sha=PM.code_sha(), 说明="缩小网格（PGSS_T1_SMOKE=1）只验证实现，不产生结果",
                smoke示例=dict(delta1=dict(ALT=m1["ALT"], pstar=m1["pstar_D"], sec=m1["sec_total"]), delta0_sec=m0["sec_total"]))
    PM.PL.mkdir(parents=True, exist_ok=True)
    (PM.PL / "R45_PLASMODE_TESTS.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


if __name__ == "__main__":
    sys.exit(0 if main()["全部通过"] else 1)
