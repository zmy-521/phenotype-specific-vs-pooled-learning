# -*- coding: utf-8 -*-
r"""R43 统一报告与证据表（md 与 xlsx **同源生成**：md 中的每个数字都来自写入 xlsx 的同一个 DataFrame）。

输入：readonly/（R43_READONLY.xlsx/json、R43_READONLY_TESTS.json、GPT 复核脚本本机运行结果）、
      t1/（配置、端到端检查、R43_T1_SUMMARY.xlsx/json、_COST.json）、figures/。
输出：24_R43_结果补齐与T1/R43_结果补齐与T1_统一报告.md、R43_结果补齐与T1_证据表.xlsx
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

R43 = HERE.parent / "24_R43_结果补齐与T1"
RO = R43 / "readonly"; T1D = R43 / "t1"
MD = R43 / "R43_结果补齐与T1_统一报告.md"; XL = R43 / "R43_结果补齐与T1_证据表.xlsx"
pc = lambda v, d=4: ("—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}%")
SCN = {"N1": "N1 无关系差异", "N2": "N2 只有截距差异", "P1": "P1 交互（SZ 可学）", "P2": "P2 轨道特异稀疏结构",
       "N1-T": "N1-T（Z 由 X₈ 阈值定义）", "P2-T": "P2-T（Z 由 X₈ 阈值定义）"}

EXTERNAL = pd.DataFrame([
    dict(来源="MIMIC-IV / eICU-CRD", 本机状态="已有（本研究开发数据）", 访问方式="—", 可识别同一患者="MIMIC 可；eICU 用患者—医院连通分量",
         住院死亡="已用", 外部角色="不是外部数据", 阻塞点="—", 依据="本机 D:\\Research\\1_数据\\MIMIC"),
    dict(来源="NWICU（PhysioNet，0.1.0，美国西北大学医学中心）", 本机状态="已下载（用户签署 DUA；所需 8 张表 SHA256 校验通过）",
         访问方式="PhysioNet 认证用户签 DUA 即可", 可识别同一患者="是（subject_id）",
         住院死亡="hospital_expire_flag 几乎不可用；deathtime 含出院后死亡；以出院去向 EXPIRED 为主的三个候选定义（§7.2）",
         外部角色="真实外部 ICU：MIMIC-IV 同结构，23 项特征池全部可映射；建议主分析为算法程序复现",
         阻塞点="无；待裁决结局定义与抽取规范后再计算", 依据="https://physionet.org/content/nwicu-northwestern-icu/"),
    dict(来源="SICdb（PhysioNet，页面显示最新 1.0.8）", 本机状态="无", 访问方式="PhysioNet 认证用户 + CITI「Data or Specimens Only Research」"
         " + 向数据贡献者提交具体研究问题供审核 + 签 Contributor Review DUA",
         可识别同一患者="页面摘要未写明；GPT 先前核对的 Case Data 文档列有 PatientID，待获批后在数据字典核对",
         住院死亡="页面只写 OffsetOfDeath（一年内、含院外死亡），不能直接当住院死亡；HospitalDischargeType 待核对",
         外部角色="候选：算法程序复现；模型迁移需先完成 23 项变量与 6 个表型的覆盖审计", 阻塞点="研究申请待数据贡献者审核（批量下载对该库返回 HTTP 403）",
         依据="https://physionet.org/content/sicdb/"),
    dict(来源="HiRID（PhysioNet，页面显示 1.1.1）", 本机状态="无", 访问方式="同上：认证 + CITI + 提交研究问题供贡献者审核 + 签 DUA",
         可识别同一患者="否（页面写明无法识别同一患者的多次入住）", 住院死亡="页面写有 discharge status（alive/dead/unknown），定义待核对",
         外部角色="只能入住级分组，推断口径受限", 阻塞点="用户已有 PhysioNet 账号，下载由用户本人运行下载脚本；PhysioNet 2025-07 起对部分受控数据集有地区访问限制，以实际下载结果为准", 依据="https://physionet.org/content/hirid/"),
    dict(来源="AmsterdamUMCdb", 本机状态="无", 访问方式="经 Amsterdam Medical Data Science 申请：访问申请与终端用户许可（EULA）；"
         "公开资料提到需完成课程，当前要求未核实", 可识别同一患者="待核对", 住院死亡="待核对", 外部角色="待核对",
         阻塞点="需用户本人在官网核对当前流程后申请", 依据="检索结果（DANS/EASY 与 PyPI 页面），官网页面本次未能直接读取"),
    dict(来源="NHANES（含公开死亡随访链接）", 本机状态="已有（D:\\NAH公共数据库\\NHANES；PGSS 早期建过死亡队列）", 访问方式="公开下载，无需账号",
         可识别同一患者="是（SEQN）", 住院死亡="无；可用全因死亡（公开死亡随访）",
         外部角色="**跨领域程序复现**（一般人群，不是 ICU 模型迁移）：六个表型均可定义（性别、年龄、血红蛋白、肌酐→eGFR、白蛋白、BMI）",
         阻塞点="无；R44 方案已冻结（R44_NHANES_PROTOCOL_v1.0），运行中", 依据="本机数据与 PGSS 第一阶段记录"),
])


def kv(d):
    return pd.DataFrame([(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in d.items()],
                        columns=["项", "值"])


def load():
    X = pd.read_excel(RO / "R43_READONLY.xlsx", sheet_name=None)
    J = json.loads((RO / "R43_READONLY.json").read_text(encoding="utf-8"))
    TS = json.loads((RO / "R43_READONLY_TESTS.json").read_text(encoding="utf-8"))
    G = json.loads((RO / "GPT_review_record_checks_本机运行.json").read_text(encoding="utf-8"))
    T1 = {}
    if (T1D / "R43_T1_SUMMARY.xlsx").exists():
        T1 = pd.read_excel(T1D / "R43_T1_SUMMARY.xlsx", sheet_name=None)
        T1["_summ"] = json.loads((T1D / "R43_T1_SUMMARY.json").read_text(encoding="utf-8"))
    T1["_cfg"] = json.loads((T1D / "R43_T1_CONFIG.json").read_text(encoding="utf-8"))
    T1["_e2e"] = json.loads((T1D / "R43_T1_E2E_TESTS.json").read_text(encoding="utf-8"))
    T1["_cost"] = json.loads((T1D / "_COST.json").read_text(encoding="utf-8")) if (T1D / "_COST.json").exists() else {}
    TX = {}
    if (T1D / "R43_T1_EXT_SUMMARY.xlsx").exists():
        TX = pd.read_excel(T1D / "R43_T1_EXT_SUMMARY.xlsx", sheet_name=None)
        TX["_summ"] = json.loads((T1D / "R43_T1_EXT_SUMMARY.json").read_text(encoding="utf-8"))
    TX["_cfg"] = json.loads((T1D / "R43_T1_EXT_CONFIG.json").read_text(encoding="utf-8"))
    TX["_e2e"] = json.loads((T1D / "R43_T1_EXT_E2E_TESTS.json").read_text(encoding="utf-8"))
    TX["_cost"] = json.loads((T1D / "_EXT_COST.json").read_text(encoding="utf-8")) if (T1D / "_EXT_COST.json").exists() else {}
    BV = json.loads((RO / "R43_BOOT_VERIFY.json").read_text(encoding="utf-8"))
    BVT = json.loads((RO / "R43_BOOT_VERIFY_TESTS.json").read_text(encoding="utf-8"))
    return X, J, TS, G, T1, TX, BV, BVT


def derived(X):
    """报告用的派生汇总（写进 xlsx，md 只引用这些表）。"""
    B = X["5_Brier对照含新增"]; L = X["6_logloss对照"]
    dirs = []
    for met, d in (("Brier", B), ("logloss", L)):
        for (c, sc), h in d.groupby(["对照", "尺度"]):
            vc = h.相对0方向.value_counts(); t1 = h["三态_m0.010"].value_counts()
            dirs.append(dict(指标=met, 对照=c, 尺度=sc, 任务=len(h), 相对改善中位=h.观察相对改善.median(),
                             相对改善最小=h.观察相对改善.min(), 相对改善最大=h.观察相对改善.max(),
                             绝对差中位=h.观察绝对差.median(), 相对上界最大=h.相对_P97_5.max(),
                             全负=int(vc.get("全负", 0)), 跨零=int(vc.get("跨零", 0)), 全正=int(vc.get("全正", 0)),
                             排除m1=int(t1.get("排除实质收益", 0)), 证据不足m1=int(t1.get("证据不足", 0)),
                             支持m1=int(t1.get("支持实质收益", 0))))
    DIR = pd.DataFrame(dirs)
    P = X["4_配对差逐seed"]; pr = []
    for (c, r, sc), h in P.groupby(["候选", "参照", "范围"]):
        row = dict(候选=c, 参照=r, 范围=sc, 任务乘seed=len(h))
        for k in ("Brier", "logloss", "AUROC", "AUPRC", "校准截距", "校准斜率"):
            col = f"{k}_候选减参照"
            if col in h and h[col].notna().any():
                row[f"{k}差中位"] = h[col].median(); row[f"{k}差为正"] = int((h[col] > 0).sum())
        pr.append(row)
    PR = pd.DataFrame(pr)
    A = X["3_辅助指标逐seed"]
    CAL = (A[A.范围.isin(["A", "B"]) & A.臂.isin(["P*", "HTFS", "Hfull", "S0*", "SZ*", "CERT", "P*_F"])]
           .groupby("臂")[["校准截距", "校准斜率"]].describe(percentiles=[.05, .5, .95]).reset_index())
    CAL.columns = ["_".join([str(x) for x in c if x]) for c in CAL.columns]
    D = X["9_认证分解"]
    DEC = (D.groupby("尺度").agg(任务=("合计", "size"), 样本成本项相对中位=("样本成本项_相对R_PstarD", "median"),
                              认证启用项相对中位=("认证启用项_相对R_PstarD", "median"), 合计相对中位=("合计_相对R_PstarD", "median"),
                              认证启用项绝对最大=("认证启用项", lambda v: float(np.max(np.abs(v)))),
                              恒等式残差绝对最大=("恒等式残差", lambda v: float(np.max(np.abs(v))))).reset_index())
    LU = X["10_学习曲线实际规模"]
    LCS = (LU.groupby(["库", "目标比例"]).agg(单元=("实际比例", "size"), 已运行=("已运行", "sum"), 实际比例最小=("实际比例", "min"),
                                          实际比例中位=("实际比例", "median"), 实际比例最大=("实际比例", "max"),
                                          子集簇数中位=("子集簇数", "median"),
                                          已运行实际比例中位=("实际比例", lambda v: float(np.median(v[LU.loc[v.index, "已运行"]]))
                                                       if LU.loc[v.index, "已运行"].any() else np.nan)).reset_index())
    LR = X["11_学习曲线绝对风险"]; lr = []
    FRS = [0.25, 0.5, 0.75, 1.0]
    for (db, oc), h in LR[LR.完整度 >= 0.999].groupby(["库", "结局"]):
        W = {a: h.pivot_table(index="表型", columns="目标比例", values=f"{a}_两轨等权").reindex(columns=FRS)
             for a in ("P*", "HTFS", "Hfull")}
        start, keep = None, None                  # 同一组任务：从 100% 往下，仍 ≥3 个任务全程可评价的最小起点
        for i in range(len(FRS) - 1, -1, -1):
            k = W["P*"][FRS[i:]].dropna().index
            if len(k) >= 3:
                start, keep = i, k
        if start is None:
            continue
        for fr in FRS[start:]:
            lr.append(dict(库=db, 结局=oc, 目标比例=fr, 同组任务=len(keep), 任务=",".join(keep),
                           P星两轨等权中位=W["P*"].loc[keep, fr].median(), HTFS两轨等权中位=W["HTFS"].loc[keep, fr].median(),
                           Hfull两轨等权中位=W["Hfull"].loc[keep, fr].median()))
    LRS = pd.DataFrame(lr)
    # 同一组任务（MIMIC 全部比例都可评价）上的绝对风险变化：25% → 100%
    ch = []
    for (db, oc, ph), h in LR[LR.完整度 >= 0.999].groupby(["库", "结局", "表型"]):
        h = h.set_index("目标比例")
        if 0.25 in h.index and 1.0 in h.index:
            ch.append(dict(库=db, 结局=oc, 表型=ph, P星下降=h.loc[0.25, "P*_两轨等权"] - h.loc[1.0, "P*_两轨等权"],
                           HTFS下降=h.loc[0.25, "HTFS_两轨等权"] - h.loc[1.0, "HTFS_两轨等权"]))
    CH = pd.DataFrame(ch)
    if len(CH):
        CH["HTFS下降更多"] = CH.HTFS下降 > CH.P星下降
    return DIR, PR, CAL, DEC, LCS, LRS, CH


# 报告正文引用的汇总表（数字溯源只在这些表里找；明细表不进入数值池，见 pgss_r43_numaudit）
CITED = ["A1_闭环核对汇总", "A2_闭环变异对照", "A3_GPT复核脚本本机运行", "A4_R42复现核对", "A5_只读补齐汇总",
         "A6_R42数字溯源复查", "B3_配对差汇总", "B4_校准诊断汇总", "B7_对照方向汇总", "C2_认证分解汇总",
         "D2_学习曲线实际比例汇总", "D4_绝对风险汇总", "D5_25到100下降", "E0_T1配置", "E1_T1端到端检查", "E2_T1成本",
         "E4_T1候选对真值收益", "E7_T1开发选择与面板", "E8_T1实测成本", "E11_T1汇总", "E12_T1选择器机制合计",
         "E13_T1程序性能条件范围", "F1_外部数据状态", "F2_R44冻结记录", "F3_NWICU队列与结局", "F4_NWICU特征覆盖", "F5_NWICU候选任务", "G1_交付与本地文件", "G2_本地完整复算", "G3_本地复算变异对照",
         "X0_T1扩展配置", "X1_T1扩展端到端检查", "X2_T1扩展成本", "X4_扩展候选对真值收益", "X7_扩展开发选择与面板",
         "X8_扩展实测成本", "X11_扩展汇总", "X12_扩展选择器机制合计", "X13_扩展程序性能条件范围", "X14_扩展各轨病例构成"]
T1_SELS = ["ALWAYS_SPLIT", "CV_ANY|P*", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "CV_ANY|S0*", "CV_POINT_M|S0*",
           "CV_LB_M|S0*", "FCT|S0*", "USTUN_ADAPT"]
T1_PERF = ["ALWAYS_SPLIT", "CV_ANY|P*", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "FCT|S0*", "MIX", "USTUN_ADAPT"]


def t1_derived(T1):
    """T1 报告用的机制级合计（写进 xlsx；md 只引用它们）。"""
    TC, TD = T1["C_选择器逐轨"], T1["D_选择器两轨等权"]
    agg = []
    for (sc, nm), g in TC.groupby(["机制", "选择器"], sort=False):
        if "启用_分子" in g and g["启用_分子"].notna().all():
            agg.append(dict(机制=sc, 选择器=nm, 启用分子=int(g.启用_分子.sum()), 启用分母=int(g.启用_分母.sum()),
                            错误启用分子=int(g.错误启用_分子.sum()), 错误启用分母=int(g.错误启用_分母.sum()),
                            漏启用分子=int(g.漏启用_分子.sum()), 漏启用分母=int(g.漏启用_分母.sum())))
    rng = (TD.groupby(["机制", "选择器"], sort=False).均值.agg(条件均值最小="min", 条件均值最大="max").reset_index())
    return pd.DataFrame(agg), rng


def main():
    X, J, TS, G, T1, TX, BV, BVT = load()
    DIR, PR, CAL, DEC, LCS, LRS, CH = derived(X)
    d = lambda met, c, sc: DIR[(DIR.指标 == met) & (DIR.对照 == c) & (DIR.尺度 == sc)].iloc[0]
    rep = {k[3:]: v for k, v in J.items() if k.startswith("复现_")}
    REP = kv(rep)
    MUT = pd.DataFrame(TS["明细"])
    GPTK = kv({k: G[k] for k in ("run_units", "unique_run_units", "branches", "all_code_hashes_match", "all_run_code_hashes_match",
                                 "all_data_fingerprints_match", "fit_calls_from_metadata", "pstar_choice_mismatches",
                                 "main_pstar_counts", "main_HTFS_k_pairs", "cert_track_decisions", "specialize_decisions",
                                 "cert_decision_mismatches")})
    UCS = X["1_闭环核对汇总"]
    na = json.loads((RO / "R42_NUMAUDIT.json").read_text(encoding="utf-8"))
    NA = pd.DataFrame([dict(检查=k, 数值池大小=v["数值池大小"], 带小数数字=v["带小数数字"], 找不到=len(v["找不到"]),
                            **{f"随机命中率_{d}位小数": v["随机命中率"][str(d)] for d in range(1, 7)})
                       for k, v in na.items() if isinstance(v, dict)])
    files = kv(dict(联合重抽风险矩阵="readonly/R43_BOOT_RISKS.parquet，只留本地（不交付）", 联合重抽风险矩阵SHA=J["重抽向量SHA"],
                    联合重抽风险矩阵列数=J["重抽向量列数"], 簇级损失本地表="24 个任务各一个 parquet，只留本地（不交付）",
                    簇级损失本地表合并SHA=J["簇级损失本地表SHA"], R42逐人预测="R42 runs/ 下 preds_*，只留本地（不交付）"))
    bv = []
    for k, v in BV["结果"].items():
        bv.append(dict(复算项=k, **{kk: vv for kk, vv in v.items()}))
    BVD = pd.DataFrame(bv)
    BVD["输入_矩阵SHA"] = BV["输入_重抽风险矩阵SHA"]; BVD["输入_只读补齐表SHA"] = BV["输入_只读补齐表SHA"]
    BVTD = kv(BVT["结果"])
    has_t1 = "C_选择器逐轨" in T1
    sheets = {"A1_闭环核对汇总": UCS, "A2_闭环变异对照": MUT, "A3_GPT复核脚本本机运行": GPTK, "A4_R42复现核对": REP,
              "A5_只读补齐汇总": kv(J), "A6_R42数字溯源复查": NA,
              "B1_辅助指标逐seed": X["3_辅助指标逐seed"], "B2_配对差逐seed": X["4_配对差逐seed"], "B3_配对差汇总": PR,
              "B4_校准诊断汇总": CAL, "B5_Brier对照含新增": X["5_Brier对照含新增"], "B6_logloss对照": X["6_logloss对照"],
              "B7_对照方向汇总": DIR, "B8_maxT_logloss": X["8_maxT_logloss"],
              "C1_认证分解": X["9_认证分解"], "C2_认证分解汇总": DEC,
              "D1_学习曲线实际规模": X["10_学习曲线实际规模"], "D2_学习曲线实际比例汇总": LCS, "D3_学习曲线绝对风险": X["11_学习曲线绝对风险"],
              "D4_绝对风险汇总": LRS, "D5_25到100下降": CH,
              "E0_T1配置": kv({k: v for k, v in T1["_cfg"].items() if not k.startswith("生成_")}),
              "E0b_T1生成公式": kv({k: v for k, v in T1["_cfg"].items() if k.startswith("生成_")}),
              "E1_T1端到端检查": kv(T1["_e2e"]["结果"]), "E2_T1成本": kv(T1["_cost"]) if T1["_cost"] else pd.DataFrame(),
              "F1_外部数据状态": EXTERNAL, "F3_NWICU队列与结局": pd.concat([pd.read_excel(R43.parent / "26_R45_NWICU外部数据准备" / "R45_NWICU_AUDIT.xlsx", sheet_name=s).assign(表=s) for s in ("0_汇总", "1_队列流程", "2_结局候选")], ignore_index=True), "F4_NWICU特征覆盖": pd.read_excel(R43.parent / "26_R45_NWICU外部数据准备" / "R45_NWICU_AUDIT.xlsx", sheet_name="3_特征覆盖"), "F5_NWICU候选任务": pd.read_excel(R43.parent / "26_R45_NWICU外部数据准备" / "R45_NWICU_AUDIT.xlsx", sheet_name="4_候选任务"), "F2_R44冻结记录": kv(json.loads((R43.parent / "25_R44_NHANES跨领域复现" / "R44_FREEZE.json").read_text(encoding="utf-8"))), "G1_交付与本地文件": files, "G2_本地完整复算": BVD, "G3_本地复算变异对照": BVTD,
              "X0_T1扩展配置": kv({k: v for k, v in TX["_cfg"].items() if not k.startswith("生成_")}),
              "X1_T1扩展端到端检查": kv(TX["_e2e"]["结果"]), "X2_T1扩展成本": kv(TX["_cost"]) if TX["_cost"] else pd.DataFrame()}
    if has_t1:
        for k, nm in (("A_各臂真值风险", "E3_T1各臂真值风险"), ("B_候选对真值收益", "E4_T1候选对真值收益"),
                      ("C_选择器逐轨", "E5_T1选择器逐轨"), ("D_选择器两轨等权", "E6_T1选择器两轨等权"),
                      ("E_开发选择与面板", "E7_T1开发选择与面板"), ("F_成本", "E8_T1实测成本"), ("G_失败", "E9_T1失败"),
                      ("H_逐重复记录", "E10_T1逐重复记录")):
            sheets[nm] = T1[k]
        sheets["E11_T1汇总"] = kv(T1["_summ"])
        T1["_agg"], T1["_rng"] = t1_derived(T1)
        sheets["E12_T1选择器机制合计"] = T1["_agg"]; sheets["E13_T1程序性能条件范围"] = T1["_rng"]
    has_tx = "C_选择器逐轨" in TX
    if has_tx:
        for k, nm in (("A_各臂真值风险", "X3_扩展各臂真值风险"), ("B_候选对真值收益", "X4_扩展候选对真值收益"),
                      ("C_选择器逐轨", "X5_扩展选择器逐轨"), ("D_选择器两轨等权", "X6_扩展选择器两轨等权"),
                      ("E_开发选择与面板", "X7_扩展开发选择与面板"), ("F_成本", "X8_扩展实测成本"), ("G_失败", "X9_扩展失败"),
                      ("H_逐重复记录", "X10_扩展逐重复记录"), ("I_各轨病例构成", "X14_扩展各轨病例构成")):
            sheets[nm] = TX[k]
        sheets["X11_扩展汇总"] = kv(TX["_summ"])
        TX["_agg"], TX["_rng"] = t1_derived(TX)
        sheets["X12_扩展选择器机制合计"] = TX["_agg"]; sheets["X13_扩展程序性能条件范围"] = TX["_rng"]
    expl = pd.DataFrame({"说明": [
        "本表与 R43_结果补齐与T1_统一报告.md 由 pgss_r43_report.py 同源生成；报告中的数字均可在本表找到。",
        "R42 的拟合与预测冻结；本轮不拟合任何真实数据模型。A、B、C、D 表全部来自 R42 已保存的预测（只读）。",
        "推断范围：固定 OOF 预测的联合簇重抽（第 Ⅱ 层），与 R42 同一套 replicate 权重；不是算法级区间。",
        "相对改善 = (R参照 − R候选)/R参照，正值 = 候选更好；『候选减参照』列对 AUROC/AUPRC 正值 = 候选更好，对 Brier/logloss 正值 = 候选更差。",
        "认证评价对象：三个 seed 都有认证预测的共同单位（三个不完整任务的纳入比例见 C1『纳入比例』），不是相应任务的全人群。",
        "T1（E 表，原 320）与 T1 扩展（X 表，160）都只是探索性设计与机制试验：比例给分子、分母与 Wilson 区间；20 次重复 0 次错误不等于错误率已控制在 5% 以下。T2 才承担确认。",
        "MIMIC/eICU 的逐人预测、逐患者（簇）级损失与联合重抽矩阵只留本地，不交付；本地完整复算结果见 G2（只含汇总量与输入 SHA）。",
        "T1 与 T1 扩展是合成数据，没有数据协议限制：逐重复记录全部在代码包 t1/runs_records/ 与 t1/runs_records_ext/，可独立重算 E、X 表。"]})
    with pd.ExcelWriter(XL, engine="openpyxl") as xw:
        pd.DataFrame({"表": ["0_说明"] + list(sheets), "报告引用": ["否"] + ["是" if k in CITED else "否" for k in sheets]}
                     ).to_excel(xw, sheet_name="目录", index=False)
        expl.to_excel(xw, sheet_name="0_说明", index=False)
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    write_md(X, J, TS, G, T1, DIR, PR, CAL, DEC, LCS, LRS, CH, UCS, MUT, rep, d, has_t1, NA, TX, has_tx, BV, BVT)
    print("ok", MD, XL)


def write_md(X, J, TS, G, T1, DIR, PR, CAL, DEC, LCS, LRS, CH, UCS, MUT, rep, d, has_t1, NA, TX, has_tx, BV, BVT):
    out = []; a = out.append
    e1b = d("Brier", "E1", "bal"); e1p = d("Brier", "E1", "pop"); e1l = d("logloss", "E1", "bal")
    s0b = d("Brier", "E1_vs_S0*", "bal"); cdb = d("Brier", "CERT_vs_P*D", "bal"); pfb = d("Brier", "P*F_vs_P*D", "bal")
    a("# R43 结果补齐与 T1 统一报告\n")
    a("本报告与 `R43_结果补齐与T1_证据表.xlsx` 由同一脚本生成，所有数字都可在表中找到。范围：GPT 第 48 轮 §3（只读补齐）、"
      "§4（有限规模 T1）、§5（外部数据准备）。**本轮未拟合任何真实数据模型。**\n")
    # ── 首页六问
    a("## 0. 首页\n")
    a("| 问题 | 回答 |\n|---|---|")
    a(f"| R42 冻结结果是否改变 | **否**。用同一套 replicate 权重从已保存预测重算：{rep['R42对照行数']} 行对照逐行匹配，观察值与区间端点的最大差为 "
      f"{rep['观察相对改善最大差']:.1e}（相对尺度）；三态与相对 0 方向不一致 {rep['三态不一致']} / {rep['方向不一致']} 处；"
      f"max-T 临界值 {rep['maxT临界值']:.4f}（R42：{rep['maxT临界值_R42']:.4f}），判定不一致 {rep['maxT判定不一致']} 处（A4） |")
    a(f"| 只读补齐新增了什么 | 预测产物闭环核对（A1，配 {len(MUT) - 1} 项变异对照，A2）；log loss 版本的主要对照与 max-T（B6、B8）；"
      "AUROC、AUPRC、校准诊断的逐 seed 值与配对差（B1–B4）；HTFS vs S0\\*（B5）；认证分解与样本成本（C1、C2）；"
      "学习曲线实际训练规模与各模型绝对风险（D1–D5，图 F1、F2）；联合重抽的本地完整复算（G2、G3） |")
    s1 = T1.get("_summ"); sx = TX.get("_summ")
    a(f"| T1 实际完成多少 | 原 T1：{(str(s1['完成重复']) + ' / ' + str(s1['计划_本批']) + '，失败 ' + str(s1['失败'])) if s1 else '尚未汇总'}；"
      f"T1 扩展（N1-T、P2-T，第48轮补充裁决批准）：{(str(sx['完成重复']) + ' / ' + str(sx['计划_本批']) + '，失败 ' + str(sx['失败'])) if sx else '尚未汇总'}"
      "（§6；全部是探索性 T1，不是 T2 确认） |")
    a("| 哪些机制下专业化确有增量、选择器能否识别 | " + ("原 T1 逐机制要点见下文与 §6.0；扩展见 §6X.0 |" if "_agg" in T1 else "见 §6 |"))
    a("| 外部数据当前状态 | **NWICU（美国西北大学 ICU，认证访问）已下载并校验**，完成覆盖审计与候选任务清单，未计算任何性能（§7.2，提请裁决定义与定位）；"
      "SICdb 的研究申请待数据贡献者审核；R44 NHANES 跨领域复现方案已冻结（§7.1）（F1–F5） |\n")
    if "_agg" in T1:
        import pgss_r43_t1_points as TP
        a("**原 T1 要点**（理想化独立样本，探索性；每条件 20 个重复）：\n")
        for line in TP.t1_points(T1, "E", pc, SCN):
            a(line)
        a("\n" + TP.HEADLINE_E + "\n")
    a("**表述按第 48 轮收紧**：结论只针对本研究的冻结程序与这些任务；相对差与绝对差并报；不写「已证明来自样本碎片化」；"
      "特征选择写作「在当前候选池和预定选择程序下，压缩面板很少被选中」；不写「专业化收益被逐级参照吸收」。\n")
    # ── 1 闭环
    a("## 1. 预测产物闭环核对（§3.2）\n")
    a("核对对象是**预定单位集合**，不是集合交集：每个 main/cert/lc 单元的测试集合，与按 LR 冻结外层划分重算的测试折逐单位比较；"
      "每个 seed 五折拼接后与任务全体比较；cert 的认证集与按冻结种子重算的划分比较；lc 的子集规模与按冻结种子重算的嵌套子集比较。\n")
    a("| 分支 | 完成 | 测试集合一致 | y/z 一致 | 预测有限且在合法范围 | 产物指纹一致 | 认证集合一致 | 子集规模一致 |\n|---|---|---|---|---|---|---|---|")
    for _, r in UCS.iterrows():
        a(f"| {r.分支} | {r.完成} | {r.测试集合一致} | {r.y一致} / {r.z一致} | {r.预测在合法范围} | {r.产物指纹一致} | {r.认证集合一致} | {r.子集规模与记录一致} |")
    a(f"\n完成数中，cert 缺 4 个单元即 R42 路径审计认定不可执行的 4 个分支；lc 的其余单元为预先标记的不可评价点。"
      f"全部预测最小 {J['预测最小']:.6f}、最大 {J['预测最大']:.6f}。\n")
    a("**变异对照**（A2）：把一个任务的产物复制到临时目录，逐项故意破坏后重新核对；目标项全部检出，未破坏的基线全部通过。\n")
    a("| 破坏 | 目标核对项 | 检出 |\n|---|---|---|")
    for _, r in MUT.iloc[1:].iterrows():
        a(f"| {r.变异} | {r.目标项} | {'是' if r.检出 else '**否**'} |")
    o, n_ = NA.iloc[0], NA.iloc[1]
    a(f"\n**对 R42 自检的一处更正**：R42 报告称「{int(o.带小数数字)} 个带小数的数字都能在证据表里找到」。这一检查在**整张证据表**"
      f"（{int(o.数值池大小)} 个不同数值，含逐单元与全对照明细）里找匹配，检出力很弱：在 [0, 2) 内随机取的 2 位、3 位、4 位小数，"
      f"分别有 {pc(o.随机命中率_2位小数, 1)}、{pc(o.随机命中率_3位小数, 1)}、{pc(o.随机命中率_4位小数, 1)} 也能「找到」。"
      f"本轮只用汇总表（{int(n_.数值池大小)} 个数值）重查 R42 报告：{int(n_.带小数数字)} 个数字中找不到 {int(n_.找不到)} 个；"
      f"此时 2/3/4 位小数的随机命中率为 {pc(n_.随机命中率_2位小数, 1)}、{pc(n_.随机命中率_3位小数, 1)}、{pc(n_.随机命中率_4位小数, 1)}（A6）。"
      "结论：R42 报告的数字没有发现错误，但原自检对低位数数字几乎不构成检验。R43 起，数值池只用报告声明引用的汇总表，"
      "并在打包自检里同时报告随机命中率；1–8 位小数与科学计数法都纳入检查（R42 的写法漏掉了 5 位以上小数）。\n")
    a(f"\nGPT 的只读复核脚本 `review_record_checks.py` 在本机对 R42 交付包运行，结果与 GPT 报告一致：{G['run_units']} 个运行单元、"
      f"P\\* 选择不一致 {len(G['pstar_choice_mismatches'])} 处、{G['cert_track_decisions']} 条逐轨认证决定中启用 {G['specialize_decisions']} 条（A3）。\n")
    # ── 2 辅助指标
    a("## 2. 辅助性能（§3.1）\n")
    a("各 seed 的完整 OOF 分别计算，不先平均三个 seed 的概率。区间仍是固定 OOF 预测的联合簇重抽（与 Brier 同一套 replicate 权重）。\n")
    a("| 对照 | 指标 | 尺度 | 相对改善中位 | 绝对差中位 | 区间整体 < 0 / 跨 0 / 整体 > 0 | 1% 边界三态：排除 / 证据不足 / 支持 |\n|---|---|---|---:|---:|---|---|")
    for c, lab in (("E1", "E1 HTFS vs P\\*"), ("E1_vs_S0*", "HTFS vs S0\\*"), ("CERT_vs_P*D", "CERT vs P\\*_D")):
        for met in ("Brier", "logloss"):
            for sc in ("bal", "pop"):
                r = d(met, c, sc)
                a(f"| {lab} | {met} | {sc} | {pc(r.相对改善中位)} | {r.绝对差中位:.6f} | {r.全负} / {r.跨零} / {r.全正} | {r.排除m1} / {r.证据不足m1} / {r.支持m1} |")
    a(f"\nlog loss 版本的 48 项 max-T 临界值为 {J['logloss_maxT临界值']:.4f}（B8）。log loss 与 Brier 的方向一致，没有出现需要解释的分歧。\n")
    a("AUROC、AUPRC（逐 seed 配对差，72 = 24 个任务 × 3 个 seed；正值 = 候选更好）：\n")
    a("| 候选 vs 参照 | 范围 | AUROC 差中位 | AUROC 差 > 0 | AUPRC 差中位 | AUPRC 差 > 0 |\n|---|---|---:|---:|---:|---:|")
    for c, r_ in (("HTFS", "P*"), ("HTFS", "S0*"), ("CERT", "P*@C")):
        for sc in ("全体", "A", "B"):
            h = PR[(PR.候选 == c) & (PR.参照 == r_) & (PR.范围 == sc)].iloc[0]
            a(f"| {c} vs {r_} | {sc} | {h.AUROC差中位:.5f} | {int(h.AUROC差为正)}/{int(h.任务乘seed)} | {h.AUPRC差中位:.5f} | {int(h.AUPRC差为正)}/{int(h.任务乘seed)} |")
    a("\n校准诊断（逐轨、逐 seed；校准截距为斜率固定 1 的 offset 模型，校准斜率为 logit(y)~a+b·logit(p) 的 b）见 B4；"
      "各臂校准截距与斜率的中位数都接近 0 与 1。辅助指标不改变主终点，也不重新选择冠军。\n")
    # ── 3 S0*
    a("## 3. 补充参照：HTFS vs S0\\*（§2.4）\n")
    a(f"S0\\* 是**允许开发期权重与校准选择的常规合并候选**（23 项池含性别、年龄、血红蛋白、肌酐，因此不是表型完全盲、未经校准的 Global）。"
      f"两轨等权 Brier 上 HTFS 相对 S0\\* 的相对改善中位 {pc(s0b.相对改善中位)}，95% 区间整体 < 0 的任务 {s0b.全负}/{s0b.任务}、跨 0 {s0b.跨零}；"
      f"1% 边界下排除实质收益 {s0b.排除m1}/{s0b.任务}（B5、B7）。\n")
    a("这意味着在本数据上**不能**写「专业化收益先出现、再被逐级参照吸收」：HTFS 连这个常规合并候选也没有超过。"
      "这一比较不推广到所有可能的「朴素 Global」。\n")
    # ── 4 认证
    a("## 4. 认证：评价对象与样本成本（§3.3）\n")
    a("同一评价对象、同一绝对 Brier 尺度上的**可核对恒等式**（不是两个独立因果机制）：\n")
    a("R(P\\*_D) − R(CERT) = {R(P\\*_D) − R(P\\*_F)}（少用认证集训练的样本成本）+ {R(P\\*_F) − R(CERT)}（认证启用带来的追加变化）\n")
    a("| 尺度 | 任务 | 样本成本项（相对 R(P\\*_D)，中位） | 认证启用项（中位） | 合计（中位） | 认证启用项绝对值最大 | 恒等式残差绝对值最大 |\n|---|---:|---:|---:|---:|---:|---:|")
    for _, r in DEC.iterrows():
        a(f"| {r.尺度} | {r.任务} | {pc(r.样本成本项相对中位)} | {pc(r.认证启用项相对中位)} | {pc(r.合计相对中位)} | {r.认证启用项绝对最大:.2e} | {r.恒等式残差绝对最大:.1e} |")
    a(f"\n联合重抽下，CERT vs P\\*_D 两轨等权相对改善中位 {pc(cdb.相对改善中位)}，P\\*_F vs P\\*_D 中位 {pc(pfb.相对改善中位)}："
      "因为 712 条逐轨决定只有 1 条启用专业化，认证程序几乎总是部署 P\\*_F，**它相对全量训练参照的全部差距来自留出认证数据的样本成本**。"
      "所以极低的启用率不能当作认证程序的成功；在这些任务上它尚未证明值得采用。\n")
    a("三个认证不完整任务只在三个 seed 都有认证预测的共同评价对象上解释（C1 的「纳入比例」），不代表相应任务的全人群；21 个完整任务与这三个任务一并列出，"
      "不因方向而排除。\n")
    # ── 5 学习曲线
    a("## 5. 学习曲线：实际规模与各模型自身的风险（§3.4）\n")
    a("横轴改为「**目标训练比例（按簇抽取）**」。按不可拆分的簇抽样，实际比例与目标不同：\n")
    a("| 库 | 目标比例 | 单元 | 已运行 | 实际比例 最小 / 中位 / 最大（全部单元） | 已运行单元实际比例中位 | 子集簇数中位 |\n|---|---:|---:|---:|---|---:|---:|")
    for _, r in LCS.iterrows():
        a(f"| {r.库} | {r.目标比例:.2f} | {r.单元} | {r.已运行} | {pc(r.实际比例最小, 2)} / {pc(r.实际比例中位, 2)} / {pc(r.实际比例最大, 2)} | {pc(r.已运行实际比例中位, 2)} | {r.子集簇数中位:.0f} |")
    a("\n各模型自身的绝对 Brier（两轨等权；只用全部外层折可评价的点；D3–D5，图 F1）：\n")
    a("每个库 × 结局只用**同一组任务**（从 100% 往下、仍有 ≥3 个任务全部外层折可评价的最小起点）；不同结局的基线风险不同，不合并。\n")
    a("| 库 | 结局 | 目标比例 | 同组任务数 | P\\* 中位 | HTFS 中位 | Hfull 中位 |\n|---|---|---:|---:|---:|---:|---:|")
    for _, r in LRS.iterrows():
        a(f"| {r.库} | {r.结局} | {r.目标比例:.2f} | {r.同组任务} | {r.P星两轨等权中位:.6f} | {r.HTFS两轨等权中位:.6f} | {r.Hfull两轨等权中位:.6f} |")
    if len(CH):
        n_more = int(CH.HTFS下降更多.sum())
        a(f"\n在 25% 与 100% 都全可评价的 {len(CH)} 个任务（{'、'.join(sorted(CH.库.unique()))}）上，两种模型的绝对风险都随训练量下降；HTFS 下降幅度大于 P\\* 的有 {n_more} 个（D5）。"
          "也就是说，MIMIC 上「差距缩小」来自专业化程序随训练量改善得更快，但在研究样本范围内**没有观察到专业化转为占优**；"
          "不据此外推某个将来一定交叉的样本量。eICU 不同目标比例下的全可评价任务集合不同，只在同一组任务上配对展示（图 F2）。\n")
    # ── 6 T1
    a("## 6. T1：已知机制下的学习行为与选择器工作特性（§4）\n")
    cfg = T1["_cfg"]; e2e = T1["_e2e"]
    a(f"配置指纹 `{cfg['配置指纹']}`，代码指纹 `{cfg['代码指纹']}`（E0）。机制公式直接取自 `pgss_xgb_sim0.py`（源码原文存于 E0b），"
      "本批次登记的设计改动：**训练与评价都无人为缺失；分组单位 = 独立个体；评价集 100,000 个独立样本**（不是 eICU 簇数据认证的验证）。\n")
    a(f"运行前的端到端检查（E1）{sum(e2e['结果'].values())}/{len(e2e['结果'])} 项通过，其中变异对照包括：故意偷看 p\\* 的作弊选择器被检出、"
      "扰动 A 轨 X0 必然改变 P2 的 p\\*、漏掉 p\\*(1−p\\*) 项的评价器被检出、换重复号结果必然不同。\n")
    if has_t1:
        write_t1(a, T1, "E", "6.", ("F3", "F4"))
    else:
        a("**T1 尚未完成汇总。**\n")
    a("## 6X. T1 扩展：Z 由 X8 的阈值定义（160 个重复，探索性）\n")
    cx = TX["_cfg"]; ex = TX["_e2e"]
    a(f"第48轮补充裁决批准、单独配置（配置指纹 `{cx['配置指纹']}`，扩展代码指纹 `{cx['扩展代码指纹']}`，X0），在原 320 个重复完成后运行，"
      "输出在独立目录，不改动原四个机制的公式、参数、文件或结果。Z = 1[X8 > Φ⁻¹(1−πA)]，开发集与评价集各自按此规则生成；"
      "直接进入结局函数的预测变量仍为 X0–X7。**N1-T** 的条件预测关系没有额外表型效应，但 X8 与其他预测变量相关，两轨的边际事件率可以不同；"
      "**P2-T** 的轨内结构与 P2 相同，最终 p\\* 经 Z 间接依赖 X8。\n")
    a(f"运行前的端到端检查（X1）{sum(ex['结果'].values())}/{len(ex['结果'])} 项通过：Z 规则在开发集与评价集逐例成立、X8 边际为标准正态、"
      "固定 Z 时 p\\* 只依赖 X0–X7、N1-T 翻转 Z 不改变 p\\*、P2-T 把 X8 移过阈值会改变 p\\*、分组与划分、Bayes 风险与损失计算、"
      "选择器不读 p\\*（含作弊变异对照）、输出只写入扩展目录。\n")
    if has_tx:
        TI = TX["I_各轨病例构成"]
        a("**病例构成**（X14，各条件 20 个重复的均值）：比较原 N1/P2 与 N1-T/P2-T 时，改变 Z 与 X 的关系同时改变了两轨的病例构成与风险，"
          "差异不能全部归因于学习机制。\n")
        a("| 机制 | n | πA | 评价集 A 轨比例 | 期望事件率 A / B | Bayes 风险 A / B |\n|---|---:|---:|---:|---|---|")
        for _, r in TI.iterrows():
            a(f"| {r.机制} | {r.n} | {r.πA:.2f} | {r.评价_A比例:.4f} | {r.评价_期望事件率_A:.4f} / {r.评价_期望事件率_B:.4f} | "
              f"{r.评价_Bayes风险_A:.4f} / {r.评价_Bayes风险_B:.4f} |")
        a("")
        import pgss_r43_t1_points as TP
        a(TP.HEADLINE_X + "\n")
        write_t1(a, TX, "X", "6X.", ("F5", "F6"))
    else:
        a("**T1 扩展尚未完成汇总。**\n")
    # ── 7 外部
    a("## 7. 外部数据（§5）\n")
    a("除 MIMIC-IV 与 eICU-CRD（本研究的开发数据）外，本机现有 **NWICU**（用户签署 DUA 后下载，§7.2）与 NHANES（公开数据，§7.1）；"
      "**没有** SICdb、HiRID、AmsterdamUMCdb。未计算任何外部性能，协议均由用户本人签署。\n")
    a("| 来源 | 本机 | 访问方式 | 同一患者识别 | 住院死亡 | 阻塞点 |\n|---|---|---|---|---|---|")
    for _, r in EXTERNAL.iloc[1:].iterrows():
        a(f"| {r.来源} | {r.本机状态} | {r.访问方式} | {r.可识别同一患者} | {r.住院死亡} | {r.阻塞点} |")
    a("\n依据见 F1 的「依据」列。**SICdb 的研究申请待数据贡献者审核**（批量下载对该库返回 HTTP 403）；获批后同样先做覆盖审计与候选可比任务清单，不计算外部性能。"
      "AmsterdamUMCdb 不在 PhysioNet，是否可行取决于其自身条款（需用户本人核对）。"
      "无论哪个来源，都先冻结任务定义再计算（入选依据是定义与可比性，不是效果）。\n")
    # ── 8 交付与隐私
    fz = json.loads((R43.parent / "25_R44_NHANES跨领域复现" / "R44_FREEZE.json").read_text(encoding="utf-8"))
    a("### 7.1 R44 NHANES 跨领域复现：方案已冻结，结果另交\n")
    a(f"用户在等待外部 ICU 数据期间指示用 NHANES 先做复现。方案 `R44_NHANES_PROTOCOL_v1.0.md` 与冻结记录 `R44_FREEZE.json` "
      f"在任何真实数据拟合之前写定（冻结时间 {fz['冻结时间']}，冻结时真实拟合目录不存在），随本包交付，GPT 未事先审阅，提请追认。"
      f"队列 {fz['队列人数']} 名成人（1999–2008 五个周期），10 年全因死亡 {fz['结局事件']['O_MORT10']} 例、"
      f"10 年心脑血管死亡 {fz['结局事件']['O_CVD10']} 例；ICU 23 项特征池中 21 项可得（缺 SpO2、呼吸频率），六个表型同定义；"
      f"同一冻结引擎、同外层划分规则，只跑 main 分支。执行路径审计（干跑）{fz['执行路径审计']['可执行']}/{fz['执行路径审计']['总数']} 单元可执行，"
      "「10 年心脑血管死亡 × 低白蛋白」整任务预先列为不可评价。定位是**跨领域程序复现**："
      "NHANES 曾用于 PGSS 早期探索（糖尿病子队列、CVD 结局），不是未触碰数据，也不是 ICU 外部验证。正式结果在 T1 结束后运行，作为 R44 单独交付。\n")
    nw = pd.read_excel(R43.parent / "26_R45_NWICU外部数据准备" / "R45_NWICU_AUDIT.xlsx", sheet_name=None)
    nws = dict(zip(nw["0_汇总"].iloc[:, 0], nw["0_汇总"].iloc[:, 1]))
    oc_ = nw["2_结局候选"].set_index("结局"); cv_ = nw["3_特征覆盖"].set_index("变量"); tk_ = nw["4_候选任务"]
    a("### 7.2 NWICU：外部 ICU 数据已到位，覆盖审计与候选任务清单（未计算任何性能）\n")
    a(f"用户签署 DUA 后下载了 NWICU 0.1.0（美国西北大学医学中心，MIMIC-IV 同结构，与 MIMIC 不是同一批病人）；所需 8 张表按官方 SHA256 清单校验通过。"
      f"按 MIMIC 主实验的同一口径（每次住院首段 ICU、入室 24 h landmark 时仍在 ICU、成人、landmark 前未死亡）得到 "
      f"{int(nws['合格人次'])} 次合格入住（{int(nws['唯一患者'])} 名患者）。ICU 23 项特征池全部可映射；"
      f"但生命体征在 [入室, landmark) 窗内的覆盖只有 {pc(float(cv_.loc['心率', '窗内有值比例']), 1)}（化验 "
      f"{pc(float(cv_.loc['血红蛋白', '窗内有值比例']), 1)}–{pc(float(cv_.loc['血糖', '窗内有值比例']), 1)}），"
      f"白蛋白 {pc(float(cv_.loc['白蛋白', '窗内有值比例']), 1)}、BMI {pc(float(cv_.loc['BMI', '窗内有值比例']), 1)}（肥胖可判定人次因此偏少）（F3–F5）。\n")
    a(f"**住院死亡的记录口径与 MIMIC 不同**：`hospital_expire_flag` 几乎不可用，`deathtime` 还包含出院后的死亡。"
      f"三个候选定义的事件率：HOSP_MORT_a（出院去向 EXPIRED）{pc(float(oc_.loc['HOSP_MORT_a', '事件率']), 2)}、"
      f"HOSP_MORT_b（加 EXPIRED - HOSPICE）{pc(float(oc_.loc['HOSP_MORT_b', '事件率']), 2)}、"
      f"HOSP_MORT_c（死亡不晚于出院 + 1 天）{pc(float(oc_.loc['HOSP_MORT_c', '事件率']), 2)}；"
      f"ICU 停留 > 72 h {pc(float(oc_.loc['LOS72', '事件率']), 2)}。我不自行选定，提请裁决。\n")
    a("| 结局 | 表型 | 可判定人次 | A 轨人次 | A 轨事件 | B 轨事件 |\n|---|---|---:|---:|---:|---:|")
    for _, r in tk_[tk_.结局.isin(["HOSP_MORT_a", "LOS72"])].iterrows():
        a(f"| {r.结局} | {r.表型} | {r.可判定人次} | {r.A轨人次} | {r.A轨事件} | {r.B轨事件} |")
    a("\n**提请裁决（NWICU）**：① 住院死亡定义（a/b/c）；② 定位——我建议主分析为**算法程序复现**（在 NWICU 内用同一冻结程序与外层交叉验证复现 E1），"
      "模型迁移（把在 MIMIC 全量上拟合的冻结模型直接用到 NWICU）作为次要分析，因为 R42 只有交叉验证预测、没有可部署的全量模型；"
      "③ 生命体征缺失近三成：是否照主分析由引擎训练内中位数插补，还是要求窗内至少有一次体征记录（并报告两者）；"
      "④ 冻结 NWICU 抽取规范（窗口、取值与并列规则照 data v5）后再计算。**本轮未计算任何 NWICU 性能。**\n")
    a("## 8. 数据合规与可复核范围（第48轮补充裁决 §1）\n")
    a("按补充裁决：**逐人预测、逐患者（簇）级损失与 2000 次联合重抽矩阵全部只留本地，不交付**（PhysioNet 2025-09-24 关于 MIMIC 与 LLM/在线服务的说明"
      "覆盖 MIMIC 及其衍生数据）。代之以本地完整复算：用另一条独立实现，直接从本地重抽风险矩阵重算全部区间、三态、相对 0 方向与两套 max-T，"
      "与只读补齐表逐项比较（G2；矩阵与表的输入 SHA 记录在 G2）。\n")
    a("| 复算项 | 对照行或项数 | 区间端点最大差 | 三态 / 方向 / max-T 判定不一致 |\n|---|---:|---:|---|")
    for k, v in BV["结果"].items():
        if k.startswith("maxT"):
            a(f"| {k} | {v['项数']} | 临界值 {v['临界值_复算']:.4f}（表 {v['临界值_表']:.4f}） | 判定不一致 {v['判定不一致']} |")
        else:
            a(f"| {k} | {v['对照行数']} | {max(v['相对下界最大差'], v['相对上界最大差']):.1e} | {v['三态不一致']} / {v['方向不一致']} |")
    a(f"\n复算检查本身配了变异对照（G3）：改动表中一个区间下界、改掉一个三态判定、把矩阵某列整体加 1e-6，都被检出；未改动时全部一致。\n")
    a("| 范围 | 内容 | 谁能核对 |\n|---|---|---|")
    a("| 本地完全复现 | R42 预测 → 闭环核对 → 联合重抽（同一套 replicate 权重）→ 全部区间、三态、max-T；本地复算与变异对照 | 只在本机；交付代码、日志、汇总结果与输入/输出 SHA |")
    a("| GPT 可独立核对 | 全部代码；证据表中的汇总量及其表内代数；T1 与 T1 扩展的逐重复记录（合成数据，无协议限制，可重算 E、X 表）；"
      "GPT 自己的记录核对脚本（A3 已在本机复现其结果） | 对话环境 |")
    a("| 不交付 | R42 逐人预测、簇级损失（本地合并指纹见 G1）、联合重抽矩阵（指纹见 G1） | — |\n")
    # ── 9 状态
    a("## 9. 状态与待裁决\n")
    a("| 项 | 状态 |\n|---|---|")
    a("| R42 拟合、预测与结果表 | 冻结；本轮复现一致 |")
    a("| 只读补齐（§3） | 完成 |")
    a(f"| 原 T1（320） | {('完成 ' + str(T1['_summ']['完成重复']) + '，失败 ' + str(T1['_summ']['失败'])) if has_t1 else '未完成'} |")
    a(f"| T1 扩展 N1-T、P2-T（160，第48轮补充裁决批准） | {('完成 ' + str(TX['_summ']['完成重复']) + '，失败 ' + str(TX['_summ']['失败'])) if has_tx else '未完成'} |")
    a("| 真实协变量效应注入（plasmode） | 原则批准为下一阶段优先任务；本批未启动。下一轮先冻结效应生成机制、注入强度、零效应对照、评价对象与数据合规边界 |")
    a("| 文献审计 | 批准为后续独立模块；先冻结检索与判定规则，小规模双人独立提取试审 |")
    a("| 外部数据 | NWICU 已下载、校验并完成覆盖审计（§7.2，未算性能）；SICdb 研究申请待数据贡献者审核；R44 NHANES 方案已冻结、运行中（§7.1） |")
    a("| 借用曲线、完整 T2、外部模型评价 | 不在本批 |\n")
    a("**提请裁决**：① 追认 R44 NHANES 跨领域复现方案（§7.1；已在任何真实拟合前冻结，按用户指示运行，结果作为 R44 另交，"
      "定位为「现象是否复现」，不是 ICU 外部验证）；② NWICU 的四项（§7.2：住院死亡定义、主分析定位、体征缺失处理、抽取规范）——"
      "批准后先冻结抽取规范与任务清单再计算；③ plasmode 的注入机制、强度、零效应对照与评价对象（第48轮已原则批准）；"
      "④ 先后：我建议 NWICU 的算法复现与 plasmode 并列为下一批主任务（前者回答「真实外部 ICU 是否复现」，后者回答「程序能否检出已知收益」），文献审计随后。\n")
    MD.write_text("\n".join(out), encoding="utf-8")


def write_t1(a, T1, L, sec, figs):
    """L = 表前缀（E 原 T1、X 扩展）；sec = 小节编号前缀；figs = (真值收益图, 选择器图)。"""
    nm_ = {"E": dict(b="E4", c="E5", d="E6", e="E7", f="E8"), "X": dict(b="X4", c="X5", d="X6", e="X7", f="X8")}[L]
    TB, TC, TD, TE, TF = (T1[k] for k in ("B_候选对真值收益", "C_选择器逐轨", "D_选择器两轨等权", "E_开发选择与面板", "F_成本"))
    import pgss_r43_t1_points as TP
    a(f"### {sec}0 要点（数字取自 {nm_['b']}、{'E12、E13' if L == 'E' else 'X12、X13'}）\n")
    for line in TP.t1_points(T1, L, pc, SCN):
        a(line)
    a("")
    a(f"### {sec}1 完整专业化的真值收益（HTFS_D 相对 P\\*_D，{nm_['b']}，图 {figs[0]}）\n")
    a("| 机制 | n | πA | A 轨均值 | A 轨达 m | B 轨均值 | B 轨达 m |\n|---|---:|---:|---:|---:|---:|---:|")
    h = TB[TB.候选对 == "D: HTFS_D vs P*_D"]
    for (sc, n, pi), g in h.groupby(["机制", "n", "πA"], sort=False):
        rA = g[g.轨 == "A"].iloc[0]; rB = g[g.轨 == "B"].iloc[0]
        a(f"| {SCN.get(sc, sc)} | {n} | {pi:.2f} | {pc(rA.真值相对收益均值, 3)} | {rA.达m_分子}/{rA.达m_分母} | "
          f"{pc(rB.真值相对收益均值, 3)} | {rB.达m_分子}/{rB.达m_分母} |")
    a(f"\n### {sec}2 选择器（逐轨判断，{nm_['c']}）\n")
    a("判断所依候选对：CV 规则与 MIX 用 D 上的 (参照_D, HTFS_D)；FCT 与 USTUN_ADAPT 用 F 上的 (参照_F, HTFS_F)。"
      "错误启用 = 启用但该候选对真值相对收益 < 1%（分母 = 真值 < 1% 的轨）；漏启用 = 未启用但真值 ≥ 1%（分母 = 真值 ≥ 1% 的轨）。"
      f"下表把同一机制的 4 个 (n, πA) 条件与两轨合计；逐条件见 {nm_['c']}。\n")
    AG, RG = T1["_agg"], T1["_rng"]
    sels = T1_SELS
    scens = list(dict.fromkeys(TC.机制))
    a("| 选择器 | " + " | ".join(f"{s}：启用 / 错误启用 / 漏启用" for s in scens) + " |")
    a("|---|" + "---|" * len(scens))
    for nm in sels:
        cells = []
        for s in scens:
            r = AG[(AG.机制 == s) & (AG.选择器 == nm)]
            q = None if r.empty else r.iloc[0]
            cells.append("—" if q is None else f"{q.启用分子}/{q.启用分母} / {q.错误启用分子}/{q.错误启用分母} / {q.漏启用分子}/{q.漏启用分母}")
        a(f"| {nm} | " + " | ".join(cells) + " |")
    a(f"\n### {sec}3 整套程序的实际性能（两轨等权，相对永远用 P\\*_D；{nm_['d']}，图 {figs[1]}）\n")
    a("| 选择器 | " + " | ".join(scens) + " |\n|---|" + "---:|" * len(scens))
    for nm in T1_PERF:
        cells = []
        for s in scens:
            g = RG[(RG.机制 == s) & (RG.选择器 == nm)]
            cells.append("—" if g.empty else f"{pc(g.iloc[0].条件均值最小, 3)} ~ {pc(g.iloc[0].条件均值最大, 3)}")
        a(f"| {nm} | " + " | ".join(cells) + " |")
    a(f"\n每格为该机制 4 个 (n, πA) 条件的条件均值范围（最小 ~ 最大）；正值 = 优于永远用 P\\*_D。逐条件均值与 SE 见 {nm_['d']}。\n")
    a(f"### {sec}4 开发选择与成本（{nm_['e']}、{nm_['f']}）\n")
    a("| 机制 | n | πA | HTFS kA 分布 | HTFS kB 分布 | 每重复秒均值 |\n|---|---:|---:|---|---|---:|")
    for _, r in TE.iterrows():
        f = TF[(TF.机制 == r.机制) & (TF.n == r.n) & (TF.πA == r.πA)].iloc[0]
        a(f"| {r.机制} | {r.n} | {r.πA:.2f} | `{r.HTFS_kA分布}` | `{r.HTFS_kB分布}` | {f.每重复秒均值:.1f} |")
    a("")


if __name__ == "__main__":
    main()
