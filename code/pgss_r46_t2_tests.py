# -*- coding: utf-8 -*-
r"""R46 T2 端到端测试（方案 §9.1；在冻结与任何正式重复之前）。每项配一个必须失败的变异对照。
缩小网格（PGSS_T1_SMOKE=1）；G4 开发集 n=2000（P2），G5–G6 用正式 n=6000；评价段大小保持正式值（覆盖 1M 评价的内存与路径）。

G1 结局函数：expit(lin_of) 与 pgss_xgb_sim0.pstar 在高斯样本上最大差 ≤ 1e-12（四机制）；sim0 源文件 SHA 等于 T1 登记值。变异：lin 加 1e-6·X0 必须超阈。
G2 重尾生成：t₅ 构造的各列方差 ≈ 1、两两相关 ≈ 0.2、X0 超额峰度 > 2；变异：高斯生成器必须不满足峰度条件。
G3 随机流：T2 与 T1 同名条件的开发集不同；8 条件 × 2 重复的开发集两两不同；同一键重复生成逐位相同。
G4 评价口径统一：强制追加到 1M 后，臂表、候选对表、选择器表在同一评价集上互相一致（相对差 ≤ 1e-12）；
   变异：bug_mixed_eval（臂与选择器只用第 0 段）必须被检出。
G5 真值隔离：打乱评价 p 后全部选择器的决定与依据量不变；变异：作弊选择器打乱前后依据量必须改变。
G6 自适应规则：不强制时，最终仍近边界项只可能出现在 1M；评价规模 ∈ {100k, 500k, 1M}；追加记录与规模一致。
G7 汇总脚本：在测试产物（真实格式写盘）上完整跑通，各表非空、列齐全；与 T1 并列表能生成。
G8 逐块预测：评价段逐块（10 万）预测与整段一次预测逐位相同（含校准的合并臂与专属臂）；变异：开发数据少一行则预测不同。
峰值内存不在本测试中记录（同一进程先后跑多项测试，峰值被前面的大数组污染）；并发由驱动在独立进程的探针与成本批上测定。
"""
import sys, os, json, pathlib, time, hashlib, shutil
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PGSS_T1_SMOKE"] = "1"
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.special import expit
import pgss_r46_t2 as T2
import pgss_xgb_sim0 as S0

TD = T2.T2DIR / "e2e"
KEYS = ["选择器", "轨", "专业化", "部署", "t", "依据1", "依据2"]
T1_SIM0_SHA = "146d59cc2e44478f"


def same(a, b, cols):
    return bool(a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True)))


def consistent(A, PR, S):
    """臂表、候选对表、选择器表是否来自同一评价集。"""
    ar = A.set_index("臂"); ok = True
    for pk, (rf, sp) in T2.PAIRS.items():
        for g in ("A", "B"):
            G_arm = (ar.loc[rf, f"R_{g}"] - ar.loc[sp, f"R_{g}"]) / ar.loc[rf, f"R_{g}"]
            G_pr = PR[(PR.候选对 == pk) & (PR.轨 == g)].G.iloc[0]
            ok &= bool(abs(G_arm - G_pr) <= 1e-12 * max(1.0, abs(G_pr)) + 1e-15)
    for _, r in S.iterrows():
        ok &= bool(abs(r.R_PstarD - ar.loc["P*_D", f"R_{r.轨}"]) <= 1e-15 + 1e-12 * r.R_PstarD)
        if r.部署 not in ("MIX", "POOLMIX"):
            ok &= bool(abs(r.R - ar.loc[r.部署, f"R_{r.轨}"]) <= 1e-15 + 1e-12 * r.R)
    return ok


def main():
    t0 = time.time(); R = {}
    T2.N_DEV = 2000
    # G1
    rng = np.random.default_rng(11); X = S0.gen_X(200_000, rng); z = rng.random(200_000) < 0.15
    mx = max(float(np.max(np.abs(expit(T2.lin_of(X, z, b)) - S0.pstar(X, z, b)))) for b in ("N1", "N2", "P1", "P2"))
    R["G1_lin_of与pstar一致"] = mx <= 1e-12
    R["G1_sim0源文件SHA等于T1登记"] = hashlib.sha256((HERE / "pgss_xgb_sim0.py").read_bytes()).hexdigest()[:16] == T1_SIM0_SHA
    mm = float(np.max(np.abs(expit(T2.lin_of(X, z, "P2") + 1e-6 * X[:, 0]) - S0.pstar(X, z, "P2"))))
    R["G1_变异_系数扰动被检出"] = mm > 1e-12
    # G2
    Xt = T2.gen_X_t(1_000_000, np.random.default_rng(12))
    v = Xt.var(axis=0); c = np.corrcoef(Xt[:, :6].T)[np.triu_indices(6, 1)]
    kurt = lambda a: float(((a - a.mean()) ** 4).mean() / a.var() ** 2 - 3)
    R["G2_t5方差约1"] = bool(np.all(np.abs(v - 1) < 0.03))
    R["G2_t5相关约0.2"] = bool(np.all(np.abs(c - 0.2) < 0.01))
    R["G2_t5重尾"] = kurt(Xt[:, 0]) > 2
    Xg = S0.gen_X(1_000_000, np.random.default_rng(13))
    R["G2_变异_高斯不满足重尾"] = not (kurt(Xg[:, 0]) > 2)
    del Xt, Xg
    # G3
    import pgss_r43_t1 as T1
    d_t1 = T1.make_data("N1", 6000, 0.15, 0)[0]["X"]
    T2.N_DEV = 6000
    r_dev = T2.streams("N1", 0)[0]; d_t2 = T2.gen("N1", 6000, r_dev)[0]
    R["G3_T2与T1开发集不同"] = not np.array_equal(d_t1, d_t2)
    firsts = []
    for cnd in T2.CONDS:
        for rep in (0, 1):
            firsts.append(T2.gen(cnd, 50, T2.streams(cnd, rep)[0])[0][0].tobytes())
    R["G3_条件与重复的开发集两两不同"] = len(set(firsts)) == len(firsts)
    R["G3_同键重复生成逐位相同"] = np.array_equal(T2.gen("P2-H", 50, T2.streams("P2-H", 3)[0])[0],
                                          T2.gen("P2-H", 50, T2.streams("P2-H", 3)[0])[0])
    T2.N_DEV = 2000
    # G4、G7：真实格式写盘（强制 1M）
    if TD.exists():
        shutil.rmtree(TD)
    runs = TD / "runs"
    m1, A1, P1, S1 = T2.run_rep("P2", 0, write=True, runs=runs, force_stages=2)
    R["G4_强制追加到1M"] = m1["n_eval"] == 1_000_000
    R["G4_臂表候选对表选择器表一致"] = consistent(A1, P1, S1)
    mb, Ab, Pb, Sb = T2.run_rep("P2", 0, write=False, force_stages=2, bug_mixed_eval=True)
    R["G4_变异_口径混用被检出"] = not consistent(Ab, Pb, Sb)
    R["G4_同一重复两次逐位一致"] = bool(A1.equals(T2.run_rep("P2", 0, write=False, force_stages=2)[1]))
    # G5（π_A=0.15 时 n=2000 的认证拟合集 A 轨过小，会在早停留出处单类别；G5–G6 用正式开发集规模 n=6000）
    T2.N_DEV = 6000
    m2, A2, P2, S2 = T2.run_rep("N1-H", 1, write=True, runs=runs)
    S2s = T2.run_rep("N1-H", 1, write=False, shuffle_pe=True)[3]
    R["G5_打乱p后选择器不变"] = same(S2, S2s, KEYS)
    c1 = T2.run_rep("N1-H", 1, write=False, cheat=True)[3]; c2 = T2.run_rep("N1-H", 1, write=False, cheat=True, shuffle_pe=True)[3]
    R["G5_变异_作弊选择器被检出"] = not same(c1, c2, KEYS)
    # G6
    ok6 = True
    for m in (m2, T2.run_rep("N2", 2, write=True, runs=runs)[0], T2.run_rep("P2-T", 3, write=True, runs=runs)[0]):
        ok6 &= m["n_eval"] in (100_000, 500_000, 1_000_000)
        ok6 &= (not m["最终仍近边界"]) or m["n_eval"] == 1_000_000
        ok6 &= len(m["追加记录"]) == m["评价段"]
        ok6 &= m["n_eval"] == sum(T2.STAGES[: m["评价段"] + 1])
    R["G6_自适应规则一致"] = bool(ok6)
    # G8 逐块预测 = 整段预测（同一开发数据上的同一确定性程序；含校准的合并臂与专属臂）
    import pgss_xgb_engine as EN
    T2.N_DEV = 2000
    rd, rev, _ = T2.streams("P2", 7)
    Xd, zd, pdv, _ = T2.gen("P2", 2000, rd); yd = (rd.random(2000) < pdv).astype(int)
    D8 = dict(X=Xd, y=yd, z=zd, g=np.arange(2000, dtype=np.int64), uid=np.arange(2000, dtype=np.int64))
    lay = EN.task_layout(Xd, zd, [f"x{i}" for i in range(Xd.shape[1])])
    EN.set_namespace(sim="R46-T2-G8")
    blocks = list(T2.eval_blocks("P2", 1, rev[1]))[:2]
    Vall = {k: (np.concatenate([b[0][k] for b in blocks]) if isinstance(blocks[0][0][k], np.ndarray) else blocks[0][0][k]) for k in blocks[0][0]}
    specs8 = [dict(arch="S0", w="none", theta=EN.GRID_POOLED[0], cal=True),
              dict(arch="HTFS", thetaA=EN.GRID_TRACK[0], thetaB=EN.GRID_TRACK[0], kA=5, kB=5, cal=True)]
    ok8 = True
    for sp in specs8:
        whole = EN.program_predict(sp, D8, Vall, lay)
        parts = np.concatenate([EN.program_predict(sp, D8, b[0], lay) for b in blocks])
        ok8 &= bool(np.array_equal(whole, parts))
    R["G8_逐块预测与整段预测逐位相同"] = ok8
    D8m = EN.sub(D8, np.arange(1, 2000))                 # 变异：开发数据少一行 ⇒ 预测必须不同
    R["G8_变异_开发数据不同则预测不同"] = not np.array_equal(EN.program_predict(specs8[0], D8m, blocks[0][0], lay),
                                                       EN.program_predict(specs8[0], D8, blocks[0][0], lay))
    R["G8_块的单位编号不重叠且连续"] = bool(np.array_equal(Vall["uid"], np.arange(len(Vall["uid"])) + 2 * 10 ** 7))
    # G7
    try:
        import pgss_r46_t2_summary as SM
        out = SM.main(runs, TD)
        need = ["TA", "TB", "TC", "TD", "TE", "TF", "TG", "TH", "TI", "TJ", "TK"]
        R["G7_汇总在测试产物上跑通"] = all(len(out[k]) > 0 for k in need)
        R["G7_与T1并列表含T1值"] = bool(out["TI"].T1均值.notna().any())
    except Exception as e:
        R["G7_汇总在测试产物上跑通"] = False; R["G7_错误"] = f"{type(e).__name__}: {e}"
    res = dict(全部通过=all(v is True for k, v in R.items() if not k.endswith("错误")), 结果=R, 秒=round(time.time() - t0),
               代码SHA={f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16]
                      for f in ("pgss_r46_t2.py", "pgss_r46_t2_summary.py", "pgss_r46_t2_tests.py")},
               每重复秒_强制1M_smoke=m1["sec_total"])
    (T2.T2DIR / "R46_T2_E2E_TESTS.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res["全部通过"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
