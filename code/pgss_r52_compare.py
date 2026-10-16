# -*- coding: utf-8 -*-
r"""R52：原冻结结果（R46 v2 推断表）与敏感性分支结果逐行并排（old / new / Δ），并按 GPT R52 指令 §D 机械判定 STOP/GO。
只读两套推断表与分支构建记录；不拟合。所有数值来自推断输出，不手改。
用法：python pgss_r52_compare.py [new_root]   （默认分支根目录；端到端试跑时传 zero_inference，预期 Δ 全为 0、判定 GO）
输出：<BR>/R52_NHANES_SCr_SENS_EVIDENCE.xlsx（试跑时写 <BR>/_dryrun/）、R52_COMPARE.json
"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r52_sens as S

OLD = {"nh44": S.R46 / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx",
       "target": S.R46 / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx"}
KEY = ["库", "结局", "表型", "对照", "参照", "候选", "尺度"]
ST = "三态_m0.010"


def new_paths(root):
    return {"nh44": root / "inference" / "R44_INFERENCE.xlsx", "target": root / "target_inference" / "R45_TARGET_INFERENCE.xlsx"}


def pair(o, n, key, cols):
    m = o[key + cols].merge(n[key + cols], on=key, how="outer", suffixes=("_旧", "_新"), indicator=True)
    assert (m._merge == "both").all(), f"旧新行不能一一对应：{m._merge.value_counts().to_dict()}"
    m = m.drop(columns="_merge")
    for c in cols:
        if pd.api.types.is_numeric_dtype(o[c]) and not pd.api.types.is_bool_dtype(o[c]):
            m[f"{c}_Δ"] = m[f"{c}_新"] - m[f"{c}_旧"]
    return m


def contrast_pairs(o, n):
    cols = ["观察相对改善", "相对_P2_5", "相对_P97_5", "观察绝对差", "R参照", "三态_m0.005", ST, "三态_m0.020", "相对0方向"]
    m = pair(o, n, KEY, cols)
    m["三态改变_m1"] = m[f"{ST}_旧"] != m[f"{ST}_新"]
    m["方向改变"] = m["相对0方向_旧"] != m["相对0方向_新"]
    m["点估计符号翻转"] = np.sign(m["观察相对改善_旧"]) != np.sign(m["观察相对改善_新"])
    hw = (m["相对_P97_5_旧"] - m["相对_P2_5_旧"]) / 2
    m["|Δ点估计|/原半宽"] = (m["观察相对改善_Δ"].abs() / hw).where(hw > 0)
    m["上界仍排除1%_新"] = m["相对_P97_5_新"] < 0.01
    m["新增支持_m1"] = (m[f"{ST}_新"] == "支持实质收益") & (m[f"{ST}_旧"] != "支持实质收益")
    return m


def main(new_root=None):
    root = S.BR if new_root is None else pathlib.Path(new_root)
    dry = new_root is not None
    out = (S.BR / "_dryrun") if dry else S.BR
    out.mkdir(parents=True, exist_ok=True)
    NP = new_paths(root)
    O = {k: pd.read_excel(v, sheet_name=None) for k, v in OLD.items()}
    N = {k: pd.read_excel(v, sheet_name=None) for k, v in NP.items()}
    B = pd.read_excel(S.BR / "R52_BUILD.xlsx", sheet_name=None)
    # —— 主对照与全部对照 ——
    BR_all = contrast_pairs(O["nh44"]["2_Brier对照"], N["nh44"]["2_Brier对照"])
    LL_all = contrast_pairs(O["nh44"]["3_logloss对照"], N["nh44"]["3_logloss对照"])
    E1 = BR_all[BR_all.对照 == "E1"].reset_index(drop=True)
    LAB3 = {"排除实质收益", "证据不足", "支持实质收益"}
    for df in (BR_all, LL_all):
        assert set(df[f"{ST}_旧"]) | set(df[f"{ST}_新"]) <= LAB3, "三态标签超出已知集合"
    MT = pair(O["nh44"]["5_E1_maxT"], N["nh44"]["5_E1_maxT"], ["库", "结局", "表型", "轨"],
              ["θ观察", "θ重抽SD", "max-T临界值", "同时下界", "同时上界", "同时判定_相对m"])
    MT["判定改变"] = MT["同时判定_相对m_旧"] != MT["同时判定_相对m_新"]
    DIRS = pair(O["nh44"]["4_对照方向汇总"], N["nh44"]["4_对照方向汇总"], ["指标", "对照", "尺度"],
                ["任务", "相对改善中位", "相对改善最小", "相对改善最大", "相对上界最大", "全负", "跨零", "全正", "排除m1", "证据不足m1", "支持m1"])
    INC = pair(O["nh44"]["1_任务纳入"], N["nh44"]["1_任务纳入"], ["结局", "表型"], ["任务人数", "A轨人数", "事件", "A轨事件", "纳入", "缺失单元"])
    AUX = pair(O["nh44"]["6_辅助指标逐seed"], N["nh44"]["6_辅助指标逐seed"], ["库", "结局", "表型", "seed", "臂", "范围", "评价对象"],
               ["n", "事件", "Brier", "logloss", "AUROC", "AUPRC", "校准截距", "校准斜率"])
    aux_rows = []      # R49 口径：三个 seed 的均值与范围
    for (oc, ph, arm, sc), g in AUX[AUX.臂.isin(["P*", "HTFS", "S0*"])].groupby(["结局", "表型", "臂", "范围"]):
        for met in ("AUROC", "AUPRC", "logloss", "校准截距", "校准斜率"):
            aux_rows.append(dict(结局=oc, 表型=ph, 臂=arm, 范围=sc, 指标=met, seed数=len(g),
                                 旧均值=g[f"{met}_旧"].mean(), 旧最小=g[f"{met}_旧"].min(), 旧最大=g[f"{met}_旧"].max(),
                                 新均值=g[f"{met}_新"].mean(), 新最小=g[f"{met}_新"].min(), 新最大=g[f"{met}_新"].max(),
                                 均值Δ=g[f"{met}_新"].mean() - g[f"{met}_旧"].mean()))
    AUXm = pd.DataFrame(aux_rows)
    # 变化幅度与 R46「删 1 人重训」扰动基准并列（同一格式：|Δ|、|Δ|/原半宽、符号翻转、三态改变）
    c6 = pd.read_excel(S.R46 / "R46_最终核验与T2_证据表.xlsx", sheet_name="C6_重训扰动汇总")
    def agg(df, label):
        a = df.观察相对改善_Δ.abs()
        return dict(来源=label, 行=len(df), 绝对变化中位=a.median(), 绝对变化最大=a.max(), 比值中位=df["|Δ点估计|/原半宽"].median(),
                    比值最大=df["|Δ点估计|/原半宽"].max(), 符号翻转=int(df.点估计符号翻转.sum()), 三态改变=int(df.三态改变_m1.sum()),
                    区间方向改变=int(df.方向改变.sum()))
    PERT = pd.DataFrame([agg(E1, "R52 E1 全部 11 任务 × 4 尺度"), agg(E1[E1.表型 == "eGFR降低"], "R52 E1 eGFR降低 任务（特征 + 分轨同时改变）"),
                         agg(E1[E1.表型 != "eGFR降低"], "R52 E1 其余 9 任务（只有特征改变）")]
                        + [dict(r) for r in c6.to_dict("records")])
    PRS = pair(O["nh44"]["7_配对差逐seed"], N["nh44"]["7_配对差逐seed"], ["库", "结局", "表型", "seed", "候选", "参照", "范围"],
               ["Brier_候选减参照", "logloss_候选减参照", "AUROC_候选减参照", "AUPRC_候选减参照", "Brier_相对改善"])
    CHO = O["nh44"]["9_开发选择分布"].merge(N["nh44"]["9_开发选择分布"], on=["项", "Pstar"], how="outer", suffixes=("_旧", "_新")).fillna(0)
    # —— L3 ——
    TB = contrast_pairs(O["target"]["2_Brier对照"], N["target"]["2_Brier对照"])
    JD = pair(O["target"]["4_预定判读"], N["target"]["4_预定判读"], ["对照", "结局", "表型", "判读尺度"], ["观察相对改善", "相对_P2_5", "相对_P97_5", "三态_m1", "预定措辞"])
    JD["三态改变"] = JD["三态_m1_旧"] != JD["三态_m1_新"]
    C1M = pair(O["target"]["5_C1_maxT"], N["target"]["5_C1_maxT"], ["结局", "表型", "尺度"], ["θ观察", "同时下界", "同时上界", "同时判定_相对m"])
    GAP = pair(O["target"]["6_差距收回"], N["target"]["6_差距收回"], ["目标", "结局", "表型", "尺度"], ["R_Pstar", "R_Pstar_q", "R_HTFS_q", "R_HTFS", "v10差距", "收回", "收回比例_相对v10差距"])
    # —— STOP/GO（GPT R52 §D）——
    bal = E1[E1.尺度 == "bal"]
    la = E1[(E1.结局 == "O_MORT10") & (E1.表型 == "低白蛋白")]
    rules = []
    def rule(code, desc, hit, det):
        rules.append(dict(规则=code, 内容=desc, 触发=bool(hit), 明细=det))
    rule("D-1", "NHANES 两轨等权 E1 的三态（m=1%）在任一任务改变", bal.三态改变_m1.any(),
         "；".join(f"{r.结局}×{r.表型}: {r[f'{ST}_旧']}→{r[f'{ST}_新']}" for _, r in bal[bal.三态改变_m1].iterrows()) or "无")
    rule("D-2a", "低白蛋白（O_MORT10）原 B 轨正结果的三态改变（E1 各尺度）", la.三态改变_m1.any(),
         "；".join(f"{r.尺度}: {r[f'{ST}_旧']}→{r[f'{ST}_新']}" for _, r in la.iterrows()))
    rule("D-2b", "L3 目标一致性预定判读（C1 11 项、D1）的三态改变", JD.三态改变.any(),
         "；".join(f"{r.对照} {r.结局}×{r.表型}: {r.三态_m1_旧}→{r.三态_m1_新}" for _, r in JD[JD.三态改变].iterrows()) or "无")
    newsup = pd.concat([E1[E1.新增支持_m1].assign(族="Brier E1"), TB[TB.新增支持_m1].assign(族="L3 Brier"),
                        LL_all[(LL_all.对照 == "E1") & LL_all.新增支持_m1].assign(族="log loss E1（辅助）")])
    sup = lambda s: s.astype(str).str.startswith("支持")        # max-T 标签为「支持（收益>m）」「排除（收益<m）」「证据不足」
    assert set(MT["同时判定_相对m_旧"]) | set(MT["同时判定_相对m_新"]) <= {"排除（收益<m）", "证据不足", "支持（收益>m）"}, "max-T 判定标签超出已知集合"
    mtn = MT[sup(MT["同时判定_相对m_新"]) & ~sup(MT["同时判定_相对m_旧"])]
    rule("D-3", "出现新的区间支持 ≥1% 增益（Brier E1 四个尺度、max-T、L3；log loss E1 作为辅助也计入）", len(newsup) + len(mtn) > 0,
         "；".join([f"{r.族} {r.结局}×{r.表型} {r.尺度}" for _, r in newsup.iterrows()] + [f"max-T {r.结局}×{r.表型} {r.轨}" for _, r in mtn.iterrows()]) or "无")
    TK = B["5_任务旧新对照"]
    unexpected = TK[TK.改变 & (TK.表型 != "eGFR降低")]
    egfr_ok = all((TK.loc[TK.表型 == "eGFR降低", f"新_{k}"] == TK.loc[TK.表型 == "eGFR降低", f"旧_{k}"]).all() for k in ("n", "事件"))
    fc = json.loads((S.BR / "R52_FOLDCHECK.json").read_text(encoding="utf-8")) if (S.BR / "R52_FOLDCHECK.json").exists() else {}
    elig = bool(INC.纳入_旧.equals(INC.纳入_新)) and unexpected.empty and egfr_ok and (dry or fc.get("可执行性一致", False))
    rule("D-4", "程序资格非预期变化（任务纳入、非 eGFR 任务的 n/事件/分轨、eGFR 任务的 n/事件、单元可执行性）", not elig,
         f"纳入集合相同={bool(INC.纳入_旧.equals(INC.纳入_新))}；非 eGFR 任务改变={len(unexpected)}；eGFR 任务 n/事件不变={egfr_ok}；单元可执行性一致={fc.get('可执行性一致', '试跑不适用')}")
    R = pd.DataFrame(rules)
    verdict = "SCIENTIFIC_DECISION_REQUIRED" if R.触发.any() else "GO（未触发任何 STOP 规则；结果为 sensitivity，不替代原主分析）"
    summ = dict(判定=verdict, 触发规则=R[R.触发].规则.tolist(), 新结果来源=str(root.relative_to(S.ROOT)), 试跑=dry,
                E1_bal_三态改变=int(bal.三态改变_m1.sum()), E1_全部尺度_三态改变=int(E1.三态改变_m1.sum()),
                E1_bal_最大绝对Δ=float(bal.观察相对改善_Δ.abs().max()), E1_全部_最大绝对Δ=float(E1.观察相对改善_Δ.abs().max()),
                全部Brier对照_三态改变=int(BR_all.三态改变_m1.sum()), 全部Brier对照行=len(BR_all),
                maxT判定改变=int(MT.判定改变.sum()), L3判读改变=int(JD.三态改变.sum()), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    fz = json.loads((S.BR / "R52_FREEZE.json").read_text(encoding="utf-8"))
    SRC = pd.DataFrame([
        dict(项="官方来源", 值="CDC/NCHS BIOPRO_D（2005–2006）：Standard creatinine (mg/dL) = -0.016 + 0.978 × uncalibrated；https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BIOPRO_D.htm"),
        dict(项="本地核对的原文", 值=str(S.NHDOC)),
        dict(项="分支", 值=S.BRANCH), dict(项="分支代码SHA（冻结）", 值=fz["代码SHA"]),
        dict(项="冻结学习程序代码指纹", 值=json.dumps(fz["冻结代码指纹"], ensure_ascii=False)),
        dict(项="原冻结数据指纹（R44 COHORT）", 值=json.dumps(json.loads((S.ORIG_DATA / "COHORT.json").read_text(encoding="utf-8"))["指纹"], ensure_ascii=False)),
        dict(项="分支数据指纹", 值=json.dumps(json.loads((S.DATA / "COHORT.json").read_text(encoding="utf-8"))["指纹"], ensure_ascii=False)),
        dict(项="R46 v2 O_CVD10 状态指纹", 值=json.loads(S.VERIFY.read_text(encoding="utf-8"))["NHANES"]["状态SHA"]),
        dict(项="旧结果表", 值="；".join(str(v.relative_to(S.ROOT)) for v in OLD.values())),
        dict(项="新结果表", 值="；".join(str(v.relative_to(S.ROOT)) for v in NP.values()))])
    run_qc = []
    for f, k in (("R52_EQUIV.json", "全部一致"), ("R52_PATHAUDIT.json", "与原版逐单元相同"), ("R52_INFER_ZERO_CHECK.json", "全部通过"),
                 ("R52_FOLDCHECK.json", "全部通过")):
        p = S.BR / f
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            run_qc.append(dict(核对=f, 结果=j.get(k), 摘要=json.dumps({kk: vv for kk, vv in j.items() if not isinstance(vv, (list, dict))}, ensure_ascii=False)[:500]))
    RQ = pd.DataFrame(run_qc)
    idx = [("0_目录", "本表目录"), ("1_来源与指纹", "官方来源、代码与数据指纹（无个人信息）"), ("2_拟合前核验", "STOP-before-fitting 核验"),
           ("3_QC锚点", "与 R51 T6 逐值核对"), ("4_血肌酐改动", "改动人数、缺失、差值分位"), ("5_eGFR迁移矩阵", "旧 A/B × 新 A/B"),
           ("6_任务n事件轨", "12 个任务旧新 n/事件/A/B/各轨事件"), ("7_运行QC", "等价、干跑审计、零修正推断、单元核对"),
           ("8_E1主对照", "11 任务 × 4 尺度：old/new/Δ、三态、上界"), ("9_maxT", "逐轨 max-T 旧新"), ("10_对照方向汇总", "各对照×尺度的计数旧新"),
           ("11_全部Brier对照", "E1–E7 等全部对照旧新"), ("12_logloss对照", "log loss 对照旧新"), ("13_辅助指标均值", "P*/HTFS 三 seed 均值旧新"),
           ("14_辅助指标逐seed", "AUROC/AUPRC/logloss/校准逐 seed 旧新"), ("15_配对差逐seed", "HTFS−P* 等配对差旧新"),
           ("16_L3预定判读", "C1（11 项）与 D1 旧新"), ("17_L3全部对照", "C1–C3、D1–D3、E1 旧新"), ("18_L3_C1maxT", "C1 max-T 旧新"),
           ("19_L3差距收回", "目标一致性收回比例旧新"), ("20_开发选择分布", "P* 选择与 HTFS 面板旧新"), ("21_任务纳入", "推断任务纳入旧新"), ("23_变化幅度与重训扰动基准", "R52 变化与 R46 删人重训扰动并列"),
           ("22_STOP_GO", "§D 机械判定")]
    sheets = {"0_目录": pd.DataFrame(idx, columns=["表", "内容"]), "1_来源与指纹": SRC, "2_拟合前核验": B["1_拟合前核验"], "3_QC锚点": B["2_QC锚点"],
              "4_血肌酐改动": B["3_血肌酐改动"], "5_eGFR迁移矩阵": B["4_eGFR迁移矩阵"], "6_任务n事件轨": TK, "7_运行QC": RQ,
              "8_E1主对照": E1, "9_maxT": MT, "10_对照方向汇总": DIRS, "11_全部Brier对照": BR_all, "12_logloss对照": LL_all,
              "13_辅助指标均值": AUXm, "14_辅助指标逐seed": AUX, "15_配对差逐seed": PRS, "16_L3预定判读": JD, "17_L3全部对照": TB,
              "18_L3_C1maxT": C1M, "19_L3差距收回": GAP, "20_开发选择分布": CHO, "21_任务纳入": INC,
              "22_STOP_GO": pd.concat([R, pd.DataFrame([dict(规则="判定", 内容=verdict, 触发=R.触发.any(), 明细="")])], ignore_index=True),
              "23_变化幅度与重训扰动基准": PERT}
    with pd.ExcelWriter(out / "R52_NHANES_SCr_SENS_EVIDENCE.xlsx", engine="openpyxl") as xw:
        for k, df in sheets.items():
            df.to_excel(xw, sheet_name=k, index=False)
    (out / "R52_COMPARE.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 120)
    print(R.to_string()); print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
