# -*- coding: utf-8 -*-
r"""R45 NHANES 目标一致性追加对照：统计汇总（只读 R44 runs 与 R45 target runs，不拟合）。方案：R45_NHANES_TARGET_PROTOCOL_v1.0.md。

臂（逐外层折的开发选择，测试时冻结）
  v1.0（R44 原样）：P*、HTFS（校准开关按两轨等权内层）
  pop ：P*_pop（按记录等权目标开发的合并参照，其选定版本）；HTFS_pop（HTFS 冻结参数/面板，校准开关按 pop 内层分数）
  B   ：P*_B、HTFS_B（只在 O_MORT10×低白蛋白）
对照（正值 = 候选更好；尺度 A、B、bal、pop 全部报告）
  C1 HTFS_pop vs P*_pop（判读尺度 pop）  C2 P*_pop vs P*（pop）  C3 HTFS vs P*_pop（pop）
  D1 HTFS_B vs P*_B（判读尺度 B）        D2 P*_B vs P*（B）      D3 HTFS vs P*_B（B）
联合簇重抽：与 R44 完全相同的注册表（11 个纳入任务的个体并集）、SeedSequence(20441)、B=2000、同一抽样循环
  ⇒ 与 R44 逐 replicate 配对；用 R44 已发表的 E1 区间做逐位复核（断言）。
max-T（C1，pop 尺度，11 项）：θ_t = (1−m)R_pop(P*_pop) − R_pop(HTFS_pop)，按重抽 SD 标准化。
预定判读（方案 §6）：C1、D1 在 m=1% 的三态 → 预定措辞；v1.0 差距被 P*_pop / P*_B 收回的比例只作描述。
另：GPT 第 49 轮 §4.2 的事后代数（A 轨保持 P*、B 轨换 HTFS 的两轨等权点估计）按 v1.0 风险复算，标为事后描述。
输出：27_R45_外部复现与机制核验/target/R45_TARGET_INFERENCE.xlsx、.json
"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r43_readonly as RO
import pgss_r44_nhanes_run as NR
import pgss_r44_nhanes_inference as NI
import pgss_r45_target as TG

OUT = TG.OUT
B_BOOT, CHUNK, ROOT_SEED = NI.B_BOOT, NI.CHUNK, NI.ROOT_SEED
M_MAIN = 0.01
CONTRASTS = [("C1", "P*_pop", "HTFS_pop"), ("C2", "P*", "P*_pop"), ("C3", "P*_pop", "HTFS"), ("E1", "P*", "HTFS")]
CONTRASTS_B = [("D1", "P*_B", "HTFS_B"), ("D2", "P*", "P*_B"), ("D3", "P*_B", "HTFS")]
JUDGE = {"C1": "pop", "C2": "pop", "C3": "pop", "E1": "bal", "D1": "B", "D2": "B", "D3": "B"}
VERDICT = {
    "C1": {"排除实质收益": "P*_pop 在人口加权尺度上追平 HTFS_pop（专属模型 ≥1% 的增量被区间排除）⇒ 支持「原始增量部分依赖开发目标」",
           "支持实质收益": "按样本比例开发后，专属模型的人口加权收益仍 >1%（不依赖开发目标即可解释的增量）",
           "证据不足": "两种解释仍未区分"},
    "D1": {"排除实质收益": "P*_B 追平 B 轨（≥1% 增量被排除）⇒ 取得这项 B 轨表现不需要 B 轨独立训练",
           "支持实质收益": "B 专属模型仍超过以 B 轨评分开发的合并参照 ⇒ 保留「额外独立训练收益」解释（不证明生物学机制）",
           "证据不足": "两种解释仍未区分"}}


def target_cols(target, oc, ph):
    """返回 {seed: DataFrame(unit_id 索引; P*_q, HTFS_q)} 与缺失说明、选择记录。"""
    per, miss, ch = {}, [], []
    for seed in NR.SEEDS:
        parts = []
        for fold in range(NR.K_OUTER):
            d = TG.RUNS / TG.tag_of(target, oc, ph, seed, fold)
            if not (d / "_DONE").exists():
                miss.append(f"{target} seed{seed} 折{fold}"); continue
            P = pd.read_parquet(d / "preds_test.parquet")
            sp = json.loads((d / "specs.json").read_text(encoding="utf-8"))
            Q = pd.read_parquet(NR.RUNS / NR.tag_of(oc, ph, seed, fold) / "preds_test.parquet")
            if not np.array_equal(Q.unit_id.to_numpy(), P.unit_id.to_numpy()):
                raise SystemExit(f"测试单位与 R44 不同：{d.name}")
            ps = sp["pstar"]; v = "cal" if sp["specs"][ps]["cal"] else "raw"
            parts.append(pd.DataFrame({"unit_id": P.unit_id, f"P*_{target}": P[f"{ps}|{v}"].to_numpy(),
                                       f"HTFS_{target}": Q[f"HTFS|{sp['htfs_chosen_target']}"].to_numpy()}))
            ch.append(dict(目标=target, 结局=oc, 表型=ph, seed=seed, 外层折=fold, Pstar_q=f"{ps}|{v}",
                           HTFS版本_q=sp["htfs_chosen_target"],
                           **{f"HTFS内层_{ver}_{k}": sp["htfs_inner"][ver][k] for ver in ("raw", "cal") for k in ("A", "B", "bal", "pop")}))
        if len(parts) == NR.K_OUTER:
            per[seed] = pd.concat(parts).set_index("unit_id").sort_index()
    return per, miss, ch


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    inc = json.loads((OUT / "INCLUDED_TASKS.json").read_text(encoding="utf-8"))["纳入任务"]
    tasks, status, CH, CH0 = {}, [], [], []
    for oc, ph in inc:
        per44, miss44, ch44 = NI.collect(oc, ph)
        assert not miss44 and len(per44) == 3, f"R44 任务 {oc}×{ph} 不完整"
        CH0 += ch44
        perq, missq, chq = target_cols("pop", oc, ph)
        CH += chq
        perB, missB = {}, []
        if (oc, ph) in TG.B_TASKS:
            perB, missB, chB = target_cols("B", oc, ph); CH += chB
        ok = len(perq) == 3 and not missq and (not tuple((oc, ph)) in TG.B_TASKS or (len(perB) == 3 and not missB))
        status.append(dict(结局=oc, 表型=ph, 纳入=ok, 缺失=len(missq) + len(missB), 说明="；".join((missq + missB)[:5])))
        if not ok:
            continue
        per = {}
        for s in NR.SEEDS:
            d = per44[s][["y", "z", "P*", "HTFS"]].join(perq[s], how="left")
            if s in perB:
                d = d.join(perB[s], how="left")
            assert d.notna().all().all(), "追加臂缺预测"
            per[s] = d
        idx = per[NR.SEEDS[0]].index
        y = per[NR.SEEDS[0]].y.to_numpy(float); z = per[NR.SEEDS[0]].z.to_numpy(bool)
        arms = [c for c in per[NR.SEEDS[0]].columns if c not in ("y", "z")]
        L = {a: np.mean([(d[a].to_numpy(float) - y) ** 2 for d in per.values()], axis=0) for a in arms}
        LL = {a: np.mean([RO.logloss_vec(y, d[a].to_numpy(float)) for d in per.values()], axis=0) for a in arms}
        tasks[(oc, ph)] = (idx.to_numpy(np.int64), y, z, L, LL)
        print(f"{oc} {ph} 读入 {time.time()-t0:.0f}s", flush=True)
    ST = pd.DataFrame(status)
    if not ST.纳入.all():
        print("有任务未纳入：", ST[~ST.纳入].to_string())
    # 联合簇重抽（与 R44 同一注册表、同一随机流、同一抽样循环）
    keys = list(tasks)
    reg = np.unique(np.concatenate([tasks[k][0] for k in keys])); Mn = len(reg)
    agg = {}
    for k in keys:
        uid, y, z, L, LL = tasks[k]
        s = np.searchsorted(reg, uid); a = {}
        for met, Ls in (("brier", L), ("logloss", LL)):
            for arm, v in Ls.items():
                for sc, msk in (("A", z), ("B", ~z)):
                    a[(met, arm, sc, "n")] = np.bincount(s[msk], minlength=Mn).astype(float)
                    a[(met, arm, sc, "s")] = np.bincount(s[msk], weights=v[msk], minlength=Mn)
        agg[k] = a
    rng = np.random.default_rng(np.random.SeedSequence(ROOT_SEED)); done = 0; reps = {}
    while done < B_BOOT:
        b = min(CHUNK, B_BOOT - done)
        draw = rng.integers(0, Mn, size=(Mn, b)); Mt = np.zeros((Mn, b))
        for j in range(b):
            Mt[:, j] = np.bincount(draw[:, j], minlength=Mn)
        for k in keys:
            a = agg[k]
            for (met, arm, sc, q) in [x for x in a if x[3] == "n" and x[2] == "A"]:
                nA = a[(met, arm, "A", "n")] @ Mt; sA = a[(met, arm, "A", "s")] @ Mt
                nB = a[(met, arm, "B", "n")] @ Mt; sB = a[(met, arm, "B", "s")] @ Mt
                for sc_, num, den in (("A", sA, nA), ("B", sB, nB), ("P", sA + sB, nA + nB)):
                    reps.setdefault(("nhanes",) + k + (met, arm, sc_), []).append(
                        np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))
        done += b
        print(f"  重抽 {done}/{B_BOOT} {time.time()-t0:.0f}s", flush=True)
    reps = {k: np.concatenate(v) for k, v in reps.items()}
    T3 = {("nhanes",) + k: v for k, v in tasks.items()}
    TBr = RO.contrast_table(T3, reps, CONTRASTS + CONTRASTS_B, "brier")
    TLl = RO.contrast_table(T3, reps, CONTRASTS + CONTRASTS_B, "logloss")
    # 与 R44 的逐位复核：E1 四个尺度的点估计与区间必须相同（同一损失、同一重抽 ⇒ 相同）
    R44T = pd.read_excel(NR.R44 / "inference" / "R44_INFERENCE.xlsx", sheet_name="2_Brier对照")
    chk = []
    for _, r in TBr[TBr.对照 == "E1"].iterrows():
        o = R44T[(R44T.结局 == r.结局) & (R44T.表型 == r.表型) & (R44T.对照 == "E1") & (R44T.尺度 == r.尺度)]
        for col in ("观察相对改善", "相对_P2_5", "相对_P97_5"):
            chk.append(dict(结局=r.结局, 表型=r.表型, 尺度=r.尺度, 列=col, R45=r[col], R44=float(o[col].iloc[0]),
                            相同=bool(np.isclose(r[col], float(o[col].iloc[0]), rtol=0, atol=1e-15))))
    CHK = pd.DataFrame(chk)
    if not CHK.相同.all():
        raise SystemExit(f"与 R44 的 E1 复核不一致 {int((~CHK.相同).sum())} 项：重抽未配对，拒绝出表")
    # max-T（C1，pop 尺度）
    th = []
    for (oc, ph), (uid, y, z, L, LL) in tasks.items():
        v = (1 - M_MAIN) * reps[("nhanes", oc, ph, "brier", "P*_pop", "P")] - reps[("nhanes", oc, ph, "brier", "HTFS_pop", "P")]
        obs = (1 - M_MAIN) * L["P*_pop"].mean() - L["HTFS_pop"].mean()
        th.append(dict(结局=oc, 表型=ph, 尺度="pop", θ观察=float(obs), θ重抽SD=float(np.nanstd(v)), _v=v))
    TH = pd.DataFrame(th)
    Z = np.vstack([(r._v - r.θ观察) / r.θ重抽SD for _, r in TH.iterrows() if r.θ重抽SD > 0])
    c95 = float(np.nanquantile(np.nanmax(np.abs(Z), axis=0), .95))
    TH["max-T临界值"] = c95; TH["同时下界"] = TH.θ观察 - c95 * TH.θ重抽SD; TH["同时上界"] = TH.θ观察 + c95 * TH.θ重抽SD
    TH["同时判定_相对m"] = np.where(TH.同时下界 > 0, "支持（收益>m）", np.where(TH.同时上界 < 0, "排除（收益<m）", "证据不足"))
    TH = TH.drop(columns="_v")
    # 预定判读
    V = []
    for cn in ("C1", "D1"):
        h = TBr[(TBr.对照 == cn) & (TBr.尺度 == JUDGE[cn])]
        for _, r in h.iterrows():
            st = r["三态_m0.010"]
            V.append(dict(对照=cn, 结局=r.结局, 表型=r.表型, 判读尺度=JUDGE[cn], 观察相对改善=r.观察相对改善,
                          相对_P2_5=r.相对_P2_5, 相对_P97_5=r.相对_P97_5, 三态_m1=st, 预定措辞=VERDICT[cn][st]))
    VD = pd.DataFrame(V)
    # 差距收回比例（描述）：(R(P*) − R(P*_q)) / (R(P*) − R(HTFS_q))，判读尺度上
    G = []
    for (oc, ph), (uid, y, z, L, LL) in tasks.items():
        for q, sc in (("pop", "pop"), ("B", "B")):
            if f"P*_{q}" not in L:
                continue
            m = np.ones(len(z), bool) if sc == "pop" else ~z
            R0, Rq, Rh, Rh0 = (float(L[a][m].mean()) for a in ("P*", f"P*_{q}", f"HTFS_{q}", "HTFS"))
            G.append(dict(目标=q, 结局=oc, 表型=ph, 尺度=sc, R_Pstar=R0, R_Pstar_q=Rq, R_HTFS_q=Rh, R_HTFS=Rh0,
                          v10差距=R0 - Rh0, 收回=R0 - Rq, 收回比例_相对v10差距=(R0 - Rq) / (R0 - Rh0) if R0 != Rh0 else np.nan))
    GP = pd.DataFrame(G)
    # GPT §4.2 事后代数（v1.0 风险）：A 轨 P*、B 轨 HTFS 的两轨等权点估计改善 = ΔR_B / (R_A(P*) + R_B(P*))
    k = ("O_MORT10", "低白蛋白")
    uid, y, z, L, LL = tasks[k]
    RA, RB = float(L["P*"][z].mean()), float(L["P*"][~z].mean()); RBh = float(L["HTFS"][~z].mean())
    POST = pd.DataFrame([dict(项="A 轨保持 P*、B 轨换 HTFS（事后组合，未经独立选择与验证）", R_A_Pstar=RA, R_B_Pstar=RB, R_B_HTFS=RBh,
                              ΔR_B=RB - RBh, 两轨等权相对改善=(RB - RBh) / (RA + RB))])
    CHD = pd.DataFrame(CH)
    CHS = (CHD.groupby(["目标", "Pstar_q"]).size().rename("单元数").reset_index())
    CHS2 = CHD.groupby(["目标", "HTFS版本_q"]).size().rename("单元数").reset_index()
    P44 = pd.DataFrame(CH0); P44 = P44[P44.apply(lambda r: [r.结局, r.表型] in inc, axis=1)]
    summ = dict(纳入任务=len(tasks), 任务总数=len(ST), 簇数=int(Mn), B=B_BOOT, C1_maxT临界值=c95,
                与R44_E1复核项=len(CHK), 复核全部相同=bool(CHK.相同.all()), 秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R45_TARGET_INFERENCE.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        ST.to_excel(xw, sheet_name="1_任务纳入", index=False)
        TBr.to_excel(xw, sheet_name="2_Brier对照", index=False)
        TLl.to_excel(xw, sheet_name="3_logloss对照", index=False)
        VD.to_excel(xw, sheet_name="4_预定判读", index=False)
        TH.to_excel(xw, sheet_name="5_C1_maxT", index=False)
        GP.to_excel(xw, sheet_name="6_差距收回", index=False)
        POST.to_excel(xw, sheet_name="7_事后代数_GPT4_2", index=False)
        CHD.to_excel(xw, sheet_name="8_开发选择逐单元", index=False)
        pd.concat([CHS.rename(columns={"Pstar_q": "选择"}).assign(项="P*_q"),
                   CHS2.rename(columns={"HTFS版本_q": "选择"}).assign(项="HTFS_q 版本"),
                   P44.groupby("Pstar").size().rename("单元数").reset_index().rename(columns={"Pstar": "选择"})
                      .assign(项="P*（v1.0，同 11 任务）", 目标="bal")]).to_excel(xw, sheet_name="9_开发选择分布", index=False)
        CHK.to_excel(xw, sheet_name="10_与R44逐位复核", index=False)
    (OUT / "R45_TARGET_INFERENCE.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float))
    print(VD.to_string())


if __name__ == "__main__":
    main()
