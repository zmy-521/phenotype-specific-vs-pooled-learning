# -*- coding: utf-8 -*-
r"""R48（GPT 第 52 轮）图源、图注要点与完整补充材料索引。只读，不拟合、不模拟、不重新计算任何模型性能。

输入：R47 冻结数字包（第 52 轮验收通过）、R42–R46 已交付结果表、R47 代码包内的 T2 逐重复记录；A1 把每个输入与交付副本逐字节比对。
本模块做的事（对应第 52 轮条目）：
  ① 图 6 加入既有 CV_LB_M|P*：F6a/F6b/F6c 原样取 R47 C3、C4、C6b 的行，不重算；
  ② NHANES 尺度标签 Population-weighted → Sample-proportion-weighted：只改 F3a 的文字列，数值列不动；
  ③ 图注要点 F0：真实数据点态区间、max-T 同时判断、模拟均值区间的区别；图内说明文字也从 F0 读取（图与图源同源）；
  ④ 有限 D 候选硬选择 oracle（O1）：从 T2 逐重复记录只读派生，事后描述；用于与 GPT 复核包逐值比对；不是全部方法的上界；
  ⑤ 完整补充材料索引 S0：R47 包内表 + 接回 R42–R46 的完整对照、辅助性能、各模型自身校准等（复制进本包的 S 表与源表逐值相同）；
     已被后续修正取代的旧版表登记为「勿引用」并写明取代者；
  ⑥ 环境依赖 E1 与路径说明 E2：从代码闭包实时扫描（第三方导入、写死的绝对路径、项目内目录），不手抄。
用法：python pgss_r48_sources.py（只打印各表规模与输入核对；xlsx/md 由 pgss_r48_report.py 同源写出）
"""
import sys, io, re, ast, json, pathlib, hashlib, zipfile, importlib.metadata as IM
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
HAND = pathlib.Path(r"D:/Research/3_对话与交接/GPT交接_PGSS")          # 只用于与交付副本逐字节比对，不参与任何计算
OUT = ROOT / "30_R48_图稿与补充材料"
R47D = ROOT / "29_R47_写作交接"
R46D = ROOT / "28_R46_最终核验与T2"
Z46 = HAND / "46" / "R46_CODE_TESTS.zip"
Z47 = HAND / "47" / "R47_CODE_AND_FIGURES.zip"
INP = {
    "R47数字包": (R47D / "R47_冻结数字包.xlsx", ("file", HAND / "47" / "R47_冻结数字包.xlsx")),
    "R42证据表": (ROOT / "23_R42_正式主实验" / "R42_正式主实验_证据表.xlsx", ("file", HAND / "42" / "R42_正式主实验_证据表.xlsx")),
    "R43证据表": (ROOT / "24_R43_结果补齐与T1" / "R43_结果补齐与T1_证据表.xlsx", ("file", HAND / "43" / "R43_结果补齐与T1_证据表.xlsx")),
    "R44证据表": (ROOT / "25_R44_NHANES跨领域复现" / "R44_NHANES跨领域复现_证据表.xlsx", ("file", HAND / "44" / "R44_NHANES跨领域复现_证据表.xlsx")),
    "R45证据表": (ROOT / "27_R45_外部复现与机制核验" / "R45_外部复现与机制核验_证据表.xlsx",
               ("file", HAND / "45" / "R45_外部复现与机制核验_证据表.xlsx")),
    "R46证据表": (R46D / "R46_最终核验与T2_证据表.xlsx", ("file", HAND / "46" / "R46_最终核验与T2_证据表.xlsx")),
    "NWICU推断v2": (R46D / "nwicu_v2" / "inference" / "R45_NWICU_INFERENCE.xlsx", ("zip", Z46, "r46/nwicu_v2/inference/R45_NWICU_INFERENCE.xlsx")),
    "NHANES推断v2": (R46D / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx", ("zip", Z46, "r46/nhanes_v2/r44v2/inference/R44_INFERENCE.xlsx")),
    "目标一致性v2": (R46D / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx", ("zip", Z46, "r46/nhanes_v2/target_v2/R45_TARGET_INFERENCE.xlsx")),
    "T2_arms": (R47D / "t2_records" / "arms.parquet", ("zip", Z47, "t2_records/arms.parquet")),
    "T2_sel": (R47D / "t2_records" / "sel.parquet", ("zip", Z47, "t2_records/sel.parquet")),
}
CONDS = ["N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H"]
SELS6 = ["ALWAYS_SPLIT", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "MIX", "POOLMIX"]      # 图 6 a、b（R47 的 5 个 + CV_LB_M|P*）
SELS_C = ["CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*"]                                    # 图 6 c：按规则「启用」的选择器
POP_OLD, POP_NEW = "Population-weighted", "Sample-proportion-weighted"
Z = 1.96                                    # T2 均值区间：均值 ± Z·SE（与 R47 pgss_r47_numbers.side_t2 相同）
_X = {}


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def delivered_bytes(key):
    ref = INP[key][1]
    return ref[1].read_bytes() if ref[0] == "file" else zipfile.ZipFile(ref[1]).read(ref[2])


def rd(key, sheet):
    k = (key, sheet)
    if k not in _X:
        _X[k] = pd.read_excel(INP[key][0], sheet_name=sheet)
    return _X[k].copy()


def rdq(key):
    if (key, None) not in _X:
        _X[(key, None)] = pd.read_parquet(INP[key][0])
    return _X[(key, None)].copy()


def side_t2(mean, se):
    if se == 0 and mean == 0:
        return "全部重复为 0"
    lo, hi = mean - Z * se, mean + Z * se
    return "区间整体 > 0" if lo > 0 else "区间整体 < 0" if hi < 0 else "区间跨 0"


# ── A：输入核对 ─────────────────────────────────────────────────
def a1_inputs():
    rows = []
    for k, (p, ref) in INP.items():
        b, rb = p.read_bytes(), delivered_bytes(k)
        where = str(ref[1].relative_to(HAND)) + ("" if ref[0] == "file" else f" ∷ {ref[2]}")
        rows.append(dict(输入=k, 本地路径=str(p.relative_to(ROOT)), 本地SHA16=sha16(b), 交付副本=where, 交付SHA16=sha16(rb), 逐字节相同=b == rb))
    return pd.DataFrame(rows)


# ── F：图源 ─────────────────────────────────────────────────────
def f_sources():
    F = {k: rd("R47数字包", k) for k in ("F1_研究设计", "F2_ICU复现", "F3a_NHANES归因_相对", "F3b_NHANES归因_绝对", "F4_学习曲线", "F5_T2机制")}
    f3 = F["F3a_NHANES归因_相对"]
    f3["Scale"] = f3["Scale"].replace({POP_OLD: POP_NEW})
    f3["Comparison"] = f3.对照.map({"E1": "Original: HTFS vs P* (P* developed for the balanced objective)",
                                    "D1": "Aligned objective (track B): HTFS_B vs P*_B",
                                    "C1": f"Aligned objective ({POP_NEW.lower()}): HTFS_pop vs P*_pop"})
    F["F3a_NHANES归因_相对"] = f3
    c3, c4, c6 = rd("R47数字包", "C3_T2两轨等权部署"), rd("R47数字包", "C4_T2逐轨部署与变差"), rd("R47数字包", "C6b_T2重复级启用错误")
    F["F6a_T2部署两轨等权"] = c3[c3.选择器.isin(SELS6)][["条件", "选择器", "两轨等权部署改善均值", "SE", "下", "上", "均值±1.96SE方向"]].reset_index(drop=True)
    F["F6b_T2逐轨实际变差"] = c4[c4.选择器.isin(SELS6)][["条件", "选择器", "轨", "部署实际变差_分子", "部署实际变差_分母", "部署实际变差_CP下",
                                                     "部署实际变差_CP上"]].reset_index(drop=True)
    F["F6c_T2启用错误"] = c6[c6.选择器.isin(SELS_C)][["条件", "选择器", "≥1轨不达标启用_分子", "≥1轨不达标启用_分母", "≥1轨不达标启用_CP下",
                                                  "≥1轨不达标启用_CP上"]].reset_index(drop=True)
    return F


# 图内说明（英文，印在图上）与图注必须包含的要点（中文，供 GPT 写图注）。图脚本从 xlsx 的 F0 表读取「图内说明」，不另写一份。
F0_ROWS = [
    ("Fig1", "设计示意", "",
     "示意图，无区间；T2 只在合成数据上运行冻结引擎与 13 个选择程序，真实数据只运行预定比较（MIMIC-IV、eICU-CRD 另有 FCT 认证分支）；"
     "Sample-proportion-weighted = 按记录等权（等于样本中的轨比例），不是调查加权的人群估计。"),
    ("Fig2", "真实数据：逐任务点态区间",
     "Bars: pointwise 95% intervals from cluster resampling of fixed out-of-fold predictions, one task at a time (refitting variation not included; "
     "not simultaneous). Study-level track-wise judgements use max-T simultaneous bands (supplementary tables), not these intervals.",
     "逐任务点态 95% 区间（固定 OOF 预测的联合簇重抽）；不含算法级重训变异；不是同时区间；全研究层面的逐轨判断用 max-T 同时带（补充表）；"
     "36 个任务共享患者、表型相互重叠，不是 36 项独立试验。"),
    ("Fig3", "真实数据：点态区间 + 绝对风险",
     "a: pointwise 95% intervals (cluster resampling of fixed out-of-fold predictions). b: absolute Brier risk (point values).",
     "a 为点态 95% 区间；L3 是看到 L2 结果后登记的单任务追加对照；Sample-proportion-weighted = 按记录等权（样本中的轨比例），"
     "不是调查加权；P*_pop、HTFS_pop 中的 pop 是冻结表中的标识符，与该尺度同义。"),
    ("Fig4", "真实数据：只有点估计",
     "Point estimates only (no intervals).",
     "只有点估计，没有区间；只画全部外层折可评价的点（eICU-CRD 在 25% 与 50% 没有这样的点）；NWICU 与 NHANES 没有学习曲线。"),
    ("Fig5", "模拟：均值区间",
     "Simulation, 200 replicates per condition: point = mean; thick bar = mean ± 1.96 SE (Monte Carlo uncertainty of the mean); "
     "thin bar = 5th–95th percentile across replicates. These are not real-data confidence intervals.",
     "均值 ± 1.96 SE 是 200 次重复均值的 Monte Carlo 不确定度，不是真实数据的置信区间；P5–P95 是重复间变异；"
     "只对所设生成机制、开发规模与 A 轨比例成立。"),
    ("Fig6", "模拟：均值区间 + 比例 CP 区间",
     "a: mean ± 1.96 SE over 200 replicates (Monte Carlo uncertainty of the mean). b, c: % of 200 replicates with Clopper–Pearson 95% CI. "
     "b counts replicates whose final track risk is worse than P*_D (by ≥ 2 integration SE) for any reason; it is not the false-enabling rate in c. "
     "FCT deploys models fitted on F (2/3 of D): HTFS_F on certified tracks, otherwise P*_F.",
     "a 同 Fig5；b、c 是 200 次重复中的比例与 Clopper–Pearson 区间（指定生成条件下的 Monte Carlo 估计，不是错误率保证）；"
     "FCT 部署 F 训练的模型（认证通过的轨部署 HTFS_F，否则 P*_F），所以 FCT 在 b 中的风险恶化含 P*_F 的样本成本；"
     "b 不是 c 的错误启用率；CV_LB_M|P* 的近似单侧下界与认证用同一统计量，但没有错误率保证。"),
]


def f0_notes():
    return pd.DataFrame(F0_ROWS, columns=["图", "区间或计量类型", "图内说明", "图注必须包含的要点"])


# ── O：有限 D 候选硬选择 oracle（事后描述）───────────────────────
ORACLE_SETS = {
    "O-2：{P*_D, HTFS_D}": ["P*_D", "HTFS_D"],
    "O-7：全部 D 臂": ["P*_D", "HTFS_D", "Hfull_D", "HGFS_D", "S0*_D", "SZ*_D", "ALT_D"],
}
ORACLE_NOTE = ("事后描述性补充：每次重复、每轨在所列 D 臂候选中取真值风险最低者（硬选择，按真值事后选），相对 P*_D 计算改善；"
               "不是 MIX（可混合预测）、FCT（部署 F 训练模型）等全部方法的性能上界；未重跑 T2；不是预定的确认性结果。")


def o1_oracle():
    a = rdq("T2_arms")
    W = a.pivot(index="重复", columns="臂", values=["R_A", "R_B"])
    refA, refB = W[("R_A", "P*_D")], W[("R_B", "P*_D")]
    ref = (refA + refB) / 2
    rows = []
    for nm, cs in ORACLE_SETS.items():
        RA, RB = W["R_A"][cs], W["R_B"][cs]
        rA, rB = RA.min(axis=1), RB.min(axis=1)
        d = pd.DataFrame({"bal": (ref - (rA + rB) / 2) / ref, "A": (refA - rA) / refA, "B": (refB - rB) / refB,
                          "A_htfs": RA.idxmin(axis=1) == "HTFS_D", "B_htfs": RB.idxmin(axis=1) == "HTFS_D"})
        d["条件"] = d.index.str.split("__").str[0]
        for c in CONDS:
            g = d[d.条件 == c]; n = len(g)
            se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
            m, s = float(g.bal.mean()), se(g.bal)
            rows.append({"候选集": nm, "条件": c, "重复": n, "两轨等权改善均值": m, "SE": s, "下": m - Z * s, "上": m + Z * s,
                         "均值±1.96SE方向": side_t2(m, s), "P5": float(g.bal.quantile(0.05)), "P95": float(g.bal.quantile(0.95)),
                         "A轨改善均值": float(g.A.mean()), "A轨SE": se(g.A), "B轨改善均值": float(g.B.mean()), "B轨SE": se(g.B),
                         "A轨取HTFS_D的重复": int(g.A_htfs.sum()), "B轨取HTFS_D的重复": int(g.B_htfs.sum())})
    o = pd.DataFrame(rows)
    o.insert(0, "性质", ORACLE_NOTE)
    return o


def a2_reference_check():
    """oracle 用的参照与改善定义必须与冻结表一致：从 sel 记录按同一公式重算 C3 两轨等权部署改善均值，与 R47 C3 比较（全部 13 个选择器）。"""
    s, a = rdq("T2_sel"), rdq("T2_arms")
    ref = a[a.臂 == "P*_D"].set_index("重复")[["R_A", "R_B"]].mean(axis=1).rename("ref")
    p = s.pivot_table(index=["重复", "选择器"], columns="轨", values="R").reset_index().merge(ref.reset_index(), on="重复")
    p["imp"] = (p.ref - (p.A + p.B) / 2) / p.ref
    p["条件"] = p.重复.str.split("__").str[0]
    mine = p.groupby(["条件", "选择器"]).imp.mean().rename("重算").reset_index()
    c3 = rd("R47数字包", "C3_T2两轨等权部署")[["条件", "选择器", "两轨等权部署改善均值"]]
    m = c3.merge(mine, on=["条件", "选择器"], how="outer", indicator=True)
    diff = (m.两轨等权部署改善均值 - m.重算).abs()
    return pd.DataFrame([dict(检查="从 T2 sel 记录按同一公式重算 C3 两轨等权部署改善均值（13 个选择器 × 8 个条件）", 行数=len(m),
                              两边都有=int((m._merge == "both").sum()), 最大绝对差=float(diff.max()),
                              通过=bool((m._merge == "both").all() and diff.max() < 1e-12))])


def a7_constants():
    """报告与图注里出现的区间常量：从代码常量读取，并核对与 R47 的定义相同（不写字面值）。"""
    import inspect, pgss_r47_numbers as N47
    same = f"mean - {Z} * se" in inspect.getsource(N47.side_t2)
    return pd.DataFrame([dict(常量="T2 均值区间的 z 值（均值 ± z·SE）", 值=Z, 来源="pgss_r48_sources.Z", 与R47定义相同=same)])


# ── S：完整补充材料索引 ─────────────────────────────────────────
G = {1: "1 数据、队列与任务", 2: "2 ICU 预定比较的完整对照（MIMIC-IV、eICU-CRD）", 3: "3 NWICU 独立复现（v2，住院死亡结局修正后）",
     4: "4 NHANES 跨领域复现（v2，O_CVD10 死因修正后）", 5: "5 NHANES 目标一致性追加对照（v2）",
     6: "6 辅助性能（AUROC、AUPRC、logloss、校准截距与斜率）", 7: "7 各模型自身校准", 8: "8 真实数据中的认证分支（FCT；MIMIC-IV、eICU-CRD）",
     9: "9 学习曲线（MIMIC-IV、eICU-CRD）", 10: "10 T2 独立模拟", 11: "11 T1 与 plasmode（探索性）", 12: "12 修正与定义敏感性",
     13: "13 方法、文献、图源与复现"}
# (组, 来源, 工作表, 处理, 英文标题, 内容, 备注, 取代者, 取代方式)
# 处理：R47包内 = 已在 R47 数字包；接回 = 复制进本包的 S 表；本轮新增；仅索引 = 过程或探索记录，不建议作补充表；已被取代 = 勿引用；缺口
IDX = [
    (1, "R47数字包", "B1_数据与任务概况", "R47包内", "Tasks: sample size, events and track composition by database", "各库任务的人数、事件与轨构成"),
    (1, "—", "MIMIC-IV/eICU-CRD 纳排流程（data v5 终版口径）", "缺口", "", "R41–R47 交付中没有单列",
     "R34 交付（34/R34_PGSS_ALL_RESULTS.xlsx 的 11_队列计数）是旧结局口径（ICU 死亡、ICU 住院 >7 天），与终版结局（住院死亡、ICU 停留 >72 h）的人数不同，不能直接引用；"
     "需要时可从冻结 data v5 只读计数（不涉及拟合），由 GPT 决定是否需要"),
    (1, "R45证据表", "E1_NWICU_队列流程", "接回", "NWICU cohort flow", "NWICU 纳排流程", "住院死亡任务另按 R46 B3 排除无生命状态证据的人次"),
    (1, "R45证据表", "E2_NWICU_结局登记", "接回", "NWICU outcome registry checks", "NWICU 结局登记与边界情况计数"),
    (1, "R45证据表", "E4_NWICU_来源适配与QC", "接回", "NWICU source mapping and variable QC", "NWICU 各变量来源、itemid、范围外剔除与并列处理"),
    (1, "R45证据表", "E5_NWICU_缺失模式", "接回", "NWICU missingness by care unit", "NWICU 按入住单元的体征缺失与住院死亡率"),
    (1, "R45证据表", "E13_NWICU_特征覆盖", "接回", "NWICU feature coverage and distributions", "NWICU 特征有值比例与分位数"),
    (1, "R46证据表", "B1_NWICU_UNKNOWN核实", "接回", "NWICU: verification of unknown discharge status", "NWICU 出院状态 UNKNOWN 的核实（住院死亡结局修正的依据）"),
    (1, "R46证据表", "B3_NWICU_住院死亡任务变化", "接回", "NWICU in-hospital death tasks before and after outcome correction", "住院死亡任务修正前后的人次与事件"),
    (1, "R44证据表", "A2_纳排流程", "接回", "NHANES cohort flow", "NHANES 纳排流程", "O_CVD10 另按 R46 C1–C2 修正 1 人"),
    (1, "R44证据表", "A3_表型与任务", "接回", "NHANES phenotype definitions and tasks", "NHANES 表型定义、切点与任务规模",
     "O_CVD10 的事件数以 R46 修正后为准（R47 B1）；O_CVD10×低白蛋白不可执行"),
    (1, "R46证据表", "C1_NHANES_未知死因", "接回", "NHANES: deaths with unknown cause by task and track", "NHANES 死因未知者所在任务与轨"),

    (2, "R47数字包", "B2_三层总结果表", "R47包内", "Primary comparison E1 by evidence layer and scale", "三层真实数据 E1 全尺度（含 C1、D1）"),
    (2, "R47数字包", "B3_分层计数", "R47包内", "Three-state and direction counts by layer, database and scale", "三态与方向计数"),
    (2, "R47数字包", "B4_maxT汇总", "R47包内", "Per-track max-T simultaneous judgements (summary)", "逐轨 max-T 同时判断汇总"),
    (2, "R43证据表", "B5_Brier对照含新增", "接回", "ICU: all pre-specified Brier contrasts", "MIMIC-IV、eICU-CRD 全部 Brier 对照（E1–E7、E1_vs_S0*、认证、P*_F vs P*_D；四个尺度；三个 m）",
     "包含 R42 C2 的全部行（R43 A4 逐位复现）"),
    (2, "R42证据表", "C2_全部对照", "已被取代", "", "R42 全部 Brier 对照", "", "R43证据表∷B5_Brier对照含新增", "包含（逐位）"),
    (2, "R43证据表", "B7_对照方向汇总", "接回", "ICU: direction and three-state summary of all contrasts", "全部对照的方向与三态汇总（Brier 与 logloss）"),
    (2, "R42证据表", "C3_对照方向汇总", "已被取代", "", "R42 对照方向汇总", "", "R43证据表∷B7_对照方向汇总", "包含（逐位）"),
    (2, "R42证据表", "C4_对照幅度汇总_bal", "接回", "ICU: magnitude of contrasts on the balanced scale", "两轨等权尺度上各对照的幅度汇总"),
    (2, "R42证据表", "C5_E1_maxT", "接回", "ICU: per-track max-T simultaneous bands (Brier)", "MIMIC-IV、eICU-CRD 逐轨 max-T 同时带（Brier）"),
    (2, "R42证据表", "C6_maxT判定汇总", "仅索引", "", "max-T 判定汇总", "R47 B4 已含同一信息"),
    (2, "R42证据表", "C1_E1三态汇总", "仅索引", "", "E1 三态汇总", "R47 B3 已含同一信息"),
    (2, "R43证据表", "B6_logloss对照", "接回", "ICU: log-loss contrasts", "MIMIC-IV、eICU-CRD logloss 对照"),
    (2, "R43证据表", "B8_maxT_logloss", "接回", "ICU: per-track max-T simultaneous bands (log loss)", "逐轨 max-T 同时带（logloss）"),
    (2, "R42证据表", "C7_eICU_t敏感性", "接回", "eICU-CRD: t-based sensitivity intervals", "eICU-CRD 有效独立簇少时的 t 区间敏感性"),
    (2, "R42证据表", "C8_开发选择分布", "接回", "ICU: selected P* configuration and HTFS panel sizes", "P* 选择与 HTFS 面板的分布"),

    (3, "NWICU推断v2", "2_Brier对照", "接回", "NWICU: all Brier contrasts", "NWICU 全部 Brier 对照（v2）"),
    (3, "NWICU推断v2", "3_logloss对照", "接回", "NWICU: log-loss contrasts", "NWICU logloss 对照（v2）"),
    (3, "NWICU推断v2", "4_对照方向汇总", "接回", "NWICU: direction summary of contrasts", "NWICU 对照方向汇总（v2）"),
    (3, "NWICU推断v2", "5_E1_maxT", "接回", "NWICU: per-track max-T simultaneous bands", "NWICU 逐轨 max-T（v2）"),
    (3, "NWICU推断v2", "9_开发选择分布", "接回", "NWICU: selected P* configuration and HTFS panel sizes", "NWICU P* 选择与 HTFS 面板分布（v2）"),
    (3, "R46证据表", "B4_NWICU_住院死亡E1修正前后", "接回", "NWICU: E1 before and after outcome correction", "住院死亡任务 E1 修正前后"),
    (3, "R45证据表", "X10_NWICU_Brier全部对照", "已被取代", "", "NWICU 全部 Brier 对照（v1）", "", "NWICU推断v2∷2_Brier对照", "修正后重算"),
    (3, "R45证据表", "X11_NWICU_logloss对照", "已被取代", "", "NWICU logloss 对照（v1）", "", "NWICU推断v2∷3_logloss对照", "修正后重算"),
    (3, "R45证据表", "E7_NWICU_E1逐任务", "已被取代", "", "NWICU E1 逐任务（v1）", "", "R47数字包∷B2_三层总结果表", "修正后重算"),
    (3, "R45证据表", "E8_NWICU_对照方向汇总", "已被取代", "", "NWICU 对照方向汇总（v1）", "", "NWICU推断v2∷4_对照方向汇总", "修正后重算"),
    (3, "R45证据表", "E9_NWICU_maxT", "已被取代", "", "NWICU max-T（v1）", "", "NWICU推断v2∷5_E1_maxT", "修正后重算"),
    (3, "R45证据表", "E11_NWICU_开发选择", "已被取代", "", "NWICU 开发选择（v1）", "", "NWICU推断v2∷9_开发选择分布", "修正后重算"),

    (4, "NHANES推断v2", "2_Brier对照", "接回", "NHANES: all Brier contrasts", "NHANES 全部 Brier 对照（v2）"),
    (4, "NHANES推断v2", "3_logloss对照", "接回", "NHANES: log-loss contrasts", "NHANES logloss 对照（v2）"),
    (4, "NHANES推断v2", "4_对照方向汇总", "接回", "NHANES: direction summary of contrasts", "NHANES 对照方向汇总（v2）"),
    (4, "NHANES推断v2", "5_E1_maxT", "接回", "NHANES: per-track max-T simultaneous bands", "NHANES 逐轨 max-T（v2）"),
    (4, "NHANES推断v2", "9_开发选择分布", "接回", "NHANES: selected P* configuration and HTFS panel sizes", "NHANES P* 选择与 HTFS 面板分布（v2）"),
    (4, "R46证据表", "C2_NHANES_O_CVD10修正前后", "接回", "NHANES: O_CVD10 contrasts before and after cause-of-death correction", "O_CVD10 修正前后"),
    (4, "R44证据表", "X1_Brier全部对照", "已被取代", "", "NHANES 全部 Brier 对照（v1）", "", "NHANES推断v2∷2_Brier对照", "修正后重算"),
    (4, "R44证据表", "B1_E1逐任务", "已被取代", "", "NHANES E1 逐任务（v1）", "", "R47数字包∷B2_三层总结果表", "修正后重算"),
    (4, "R44证据表", "B2_对照方向汇总", "已被取代", "", "NHANES 对照方向汇总（v1）", "", "NHANES推断v2∷4_对照方向汇总", "修正后重算"),
    (4, "R44证据表", "B3_E1_maxT", "已被取代", "", "NHANES max-T（v1）", "", "NHANES推断v2∷5_E1_maxT", "修正后重算"),
    (4, "R44证据表", "B4_logloss对照", "已被取代", "", "NHANES logloss（v1）", "", "NHANES推断v2∷3_logloss对照", "修正后重算"),
    (4, "R44证据表", "C2_开发选择分布", "已被取代", "", "NHANES 开发选择（v1）", "", "NHANES推断v2∷9_开发选择分布", "修正后重算"),
    (4, "R44证据表", "D1_事后诊断_合并候选", "仅索引", "", "R44 结果知情后的合并候选诊断（v1）", "已由 R45/R46 目标一致性追加对照处理"),
    (4, "R44证据表", "D3_跨库_A轨占比", "仅索引", "", "跨库 A 轨占比（探索）"),

    (5, "R47数字包", "B5_NHANES低白蛋白归因", "R47包内", "NHANES low albumin: original and objective-aligned contrasts", "低白蛋白：E1、C1–C3、D1–D3"),
    (5, "R47数字包", "B6_NHANES差距收回", "R47包内", "NHANES low albumin: absolute Brier and gap recovered", "低白蛋白：绝对 Brier 与差距收回"),
    (5, "目标一致性v2", "2_Brier对照", "接回", "Objective-alignment control: all Brier contrasts", "追加对照全部 Brier 对照（11 个任务；v2）"),
    (5, "目标一致性v2", "3_logloss对照", "接回", "Objective-alignment control: log-loss contrasts", "追加对照 logloss（v2）"),
    (5, "目标一致性v2", "4_预定判读", "接回", "Objective-alignment control: pre-registered readings", "追加对照的预定判读（v2）"),
    (5, "目标一致性v2", "5_C1_maxT", "接回", "Objective-alignment control: max-T for C1", "C1 的 max-T（v2）"),
    (5, "目标一致性v2", "6_差距收回", "接回", "Objective-alignment control: gap recovered, all tasks", "差距收回（全部任务；v2）"),
    (5, "目标一致性v2", "9_开发选择分布", "接回", "Objective-alignment control: development selections", "追加对照的开发选择分布（v2）"),
    (5, "目标一致性v2", "7_事后代数_GPT4_2", "仅索引", "", "事后代数核对"),
    (5, "R45证据表", "D2_目标一致性_对照", "已被取代", "", "追加对照全部对照（v1）", "", "目标一致性v2∷2_Brier对照", "修正后重算"),
    (5, "R45证据表", "D1_目标一致性_预定判读", "已被取代", "", "追加对照预定判读（v1）", "", "目标一致性v2∷4_预定判读", "修正后重算"),
    (5, "R45证据表", "D4_C1_maxT", "已被取代", "", "C1 max-T（v1）", "", "目标一致性v2∷5_C1_maxT", "修正后重算"),
    (5, "R45证据表", "D5_差距收回", "已被取代", "", "差距收回（v1）", "", "目标一致性v2∷6_差距收回", "修正后重算"),
    (5, "R45证据表", "D7_开发选择分布", "已被取代", "", "开发选择分布（v1）", "", "目标一致性v2∷9_开发选择分布", "修正后重算"),
    (5, "R45证据表", "D8_logloss", "已被取代", "", "logloss（v1）", "", "目标一致性v2∷3_logloss对照", "修正后重算"),

    (6, "R43证据表", "B1_辅助指标逐seed", "接回", "ICU: auxiliary metrics by seed", "MIMIC-IV、eICU-CRD 各臂逐 seed 的 Brier、logloss、AUROC、AUPRC、平均预测、校准截距与斜率"),
    (6, "R43证据表", "B2_配对差逐seed", "接回", "ICU: paired differences in auxiliary metrics by seed", "候选减参照的逐 seed 配对差"),
    (6, "R43证据表", "B3_配对差汇总", "接回", "ICU: summary of paired differences across tasks and seeds", "配对差跨任务与 seed 汇总"),
    (6, "NWICU推断v2", "6_辅助指标逐seed", "接回", "NWICU: auxiliary metrics by seed", "NWICU 各臂逐 seed 辅助指标（v2）"),
    (6, "NWICU推断v2", "7_配对差逐seed", "接回", "NWICU: paired differences in auxiliary metrics by seed", "NWICU 逐 seed 配对差（v2）"),
    (6, "NHANES推断v2", "6_辅助指标逐seed", "接回", "NHANES: auxiliary metrics by seed", "NHANES 各臂逐 seed 辅助指标（v2）"),
    (6, "NHANES推断v2", "7_配对差逐seed", "接回", "NHANES: paired differences in auxiliary metrics by seed", "NHANES 逐 seed 配对差（v2）"),
    (6, "—", "NWICU、NHANES 辅助指标跨 seed 汇总", "缺口", "", "v2 只有逐 seed 表",
     "v1 汇总（R45 E10、R44 C1）已被取代；如需与 ICU（R43 B3）同格式的汇总，可由逐 seed 表直接汇总（不涉及拟合），由 GPT 决定是否需要"),
    (6, "R45证据表", "E10_NWICU_辅助指标差", "已被取代", "", "NWICU 辅助指标差（v1）", "", "NWICU推断v2∷7_配对差逐seed", "修正后重算"),
    (6, "R45证据表", "X12_NWICU_辅助指标逐seed", "已被取代", "", "NWICU 辅助指标逐 seed（v1）", "", "NWICU推断v2∷6_辅助指标逐seed", "修正后重算"),
    (6, "R44证据表", "C1_辅助指标差", "已被取代", "", "NHANES 辅助指标差（v1）", "", "NHANES推断v2∷7_配对差逐seed", "修正后重算"),
    (6, "R44证据表", "X2_辅助指标逐seed", "已被取代", "", "NHANES 辅助指标逐 seed（v1）", "", "NHANES推断v2∷6_辅助指标逐seed", "修正后重算"),
    (6, "R44证据表", "X3_配对差逐seed", "已被取代", "", "NHANES 配对差逐 seed（v1）", "", "NHANES推断v2∷7_配对差逐seed", "修正后重算"),

    (7, "R46证据表", "E4_各模型自身校准_摘要", "接回", "Own calibration of each model (summary)", "各库 HTFS、P* 及追加臂的自身校准（全体范围）"),
    (7, "R46证据表", "X1_各模型自身校准_全表", "接回", "Own calibration of each model (all arms and scopes)", "全部臂 × A、B、全体的自身校准"),
    (7, "R43证据表", "B4_校准诊断汇总", "接回", "ICU: distribution of calibration intercept and slope by arm", "ICU 各臂校准截距与斜率的分布"),

    (8, "R42证据表", "D0_认证纳入", "接回", "Certification branch: included tasks", "认证分支纳入任务"),
    (8, "R42证据表", "D1_认证汇总", "接回", "Certification branch: tracks specialised", "认证后专业化的轨数"),
    (8, "R42证据表", "D2_认证交叉表_描述", "接回", "Certification branch: decision versus test-fold result (descriptive)", "认证决定与测试折结果的交叉表（描述）"),
    (8, "R42证据表", "D3_认证逐单元", "接回", "Certification branch: unit-level decisions", "认证逐单元记录"),
    (8, "R43证据表", "C1_认证分解", "接回", "Certification branch: sample-size cost and enabling decomposition", "认证程序相对 P*_D 的差距分解",
     "措辞按 R45 §2.3：差距「主要」对应样本成本，认证启用项并非处处为零"),
    (8, "R43证据表", "C2_认证分解汇总", "接回", "Certification branch: decomposition summary", "分解汇总"),

    (9, "R42证据表", "E1_学习曲线", "接回", "Learning curves: relative improvement by training fraction", "学习曲线逐点（R47 F4 的来源之一）"),
    (9, "R42证据表", "E2_学习曲线汇总", "接回", "Learning curves: summary", "学习曲线汇总"),
    (9, "R43证据表", "D2_学习曲线实际比例汇总", "接回", "Learning curves: realised training fractions", "实际训练比例汇总"),
    (9, "R43证据表", "D3_学习曲线绝对风险", "接回", "Learning curves: absolute Brier risk", "各比例的绝对 Brier"),
    (9, "R43证据表", "D4_绝对风险汇总", "接回", "Learning curves: absolute risk summary", "绝对风险汇总"),
    (9, "R43证据表", "D5_25到100下降", "接回", "Learning curves: change from 25% to 100%", "25% 到 100% 的风险下降"),
    (9, "R43证据表", "D1_学习曲线实际规模", "仅索引", "", "逐单元实际规模"),
    (9, "R42证据表", "A5_学习曲线可评价", "仅索引", "", "可评价性核对"),

    (10, "R47数字包", "C1_T2条件与设计", "R47包内", "T2: conditions and design", "T2 条件、生成与评价规模"),
    (10, "R47数字包", "C2_T2候选对真值收益", "R47包内", "T2: true gain of candidate pairs", "四个候选对 × 两轨"),
    (10, "R47数字包", "C3_T2两轨等权部署", "R47包内", "T2: balanced deployed improvement, all selectors", "全部 13 个选择器"),
    (10, "R47数字包", "C4_T2逐轨部署与变差", "R47包内", "T2: track-wise deployed improvement and worsening", "全部选择器 × 两轨"),
    (10, "R47数字包", "C5_MIX等三面并列", "R47包内", "T2: average, track-wise worsening and variability side by side", "平均、逐轨变差与变异"),
    (10, "R47数字包", "C5b_MIX与POOLMIX配对差", "R47包内", "T2: MIX minus POOLMIX (paired)", "同重复配对差"),
    (10, "R47数字包", "C6a_T2逐轨启用与检出", "R47包内", "T2: track-wise enabling, false enabling and detection", "认证与 CV 规则逐轨"),
    (10, "R47数字包", "C6b_T2重复级启用错误", "R47包内", "T2: replicates enabling at least one below-margin track", "重复级至少一轨"),
    (10, "R47数字包", "C6c_T2认证样本成本", "R47包内", "T2: sample-size cost of certification (P*_F vs P*_D)", "P*_F 相对 P*_D"),
    (10, "R47数字包", "C8_两两排序反转_描述", "R47包内", "T1/T2 pairwise ranking reversals (descriptive)", "描述性"),
    (10, "R47数字包", "C9_T2极端值_初始10万", "R47包内", "T2: generator diagnostics (first 100,000 evaluation draws)", "生成分布诊断"),
    (10, "本包", "O1_有限D候选硬选择oracle", "本轮新增", "Post hoc: finite D-candidate hard-selection oracle (descriptive)",
     "每次重复、每轨在所列 D 臂候选中按真值取最低者", "事后描述；不是全部方法的上界；与 GPT 复核包的比对见报告"),
    (10, "R46证据表", "X5_T2两两排序全表", "接回", "T2: all pairwise selector comparisons (descriptive)", "全部选择器对的同重复配对差（描述）"),
    (10, "R47代码包", "t2_records/（arms、pairs、sel、rep.jsonl）", "R47包内", "T2: per-replicate records", "逐重复记录（parquet 与逐位保真 CSV）"),
    (10, "R46证据表", "D11_T2预定对照Q1至Q5", "仅索引", "", "预定对照 Q1–Q5 计数", "第 51 轮：不以 Q1–Q5 计数概括 T1 与 T2，改引 R47 C7"),
    (10, "R46证据表", "D10_T2成本与失败", "仅索引", "", "计算成本与失败"),
    (10, "R46证据表", "D12_T2调度核对", "仅索引", "", "调度核对"),

    (11, "R47数字包", "C7_T1与T2数值并列", "R47包内", "T1 (exploratory) and T2 (adopted) side by side", "T1 与 T2 数值逐项并列"),
    (11, "R47数字包", "C10_plasmode探索", "R47包内", "Plasmode simulation (exploratory)", "plasmode（探索性）"),
    (11, "R43证据表", "E3_T1各臂真值风险", "仅索引", "", "T1 各臂真值风险（探索）", "T1 的计数口径按 R45 §2 更正（R45 B1–B4、X1–X3）"),
    (11, "R43证据表", "E6_T1选择器两轨等权", "仅索引", "", "T1 选择器两轨等权（探索）"),
    (11, "R45证据表", "X1_R43_候选对_最终truth", "仅索引", "", "T1 候选对达标数（按最终 truth 更正）"),
    (11, "R45证据表", "X2_R43_选择器逐轨_更正", "仅索引", "", "T1 选择器逐轨（更正）"),
    (11, "R45证据表", "X3_R43_重复级错误启用全表", "仅索引", "", "T1 重复级错误启用（更正）"),
    (11, "R45证据表", "F2_plasmode_候选对真值收益", "仅索引", "", "plasmode 候选对真值收益（探索）", "R45 A3 登记冻结后修订 1 处（汇总脚本列名）"),
    (11, "R45证据表", "F4_plasmode_两轨等权部署", "仅索引", "", "plasmode 两轨等权部署（探索）"),

    (12, "R46证据表", "C5_重训扰动幅度", "接回", "Effect of outcome corrections on estimates", "结局修正前后的估计变化"),
    (12, "R46证据表", "C6_重训扰动汇总", "接回", "Effect of outcome corrections: summary", "修正扰动汇总"),
    (12, "R46证据表", "E3_NWICU体征两口径", "接回", "NWICU: vital-sign availability under two definitions", "NWICU 体征可得性两种口径"),
    (12, "R46证据表", "E1_预定判读范围更正", "仅索引", "", "预定判读范围更正"),
    (12, "R46证据表", "E2_R43行号更正", "仅索引", "", "R43 行号更正"),
    (12, "R45证据表", "A3_冻结后修订", "仅索引", "", "冻结后修订登记"),

    (13, "R47数字包", "D3_方法定义", "R47包内", "Definitions of arms, selectors, estimands and margins", "方法定义唯一来源（从冻结代码实时读取）"),
    (13, "R47数字包", "D4_文献比较", "R47包内", "Directly related studies (verbatim excerpts)", "文献比较"),
    (13, "R47数字包", "D7_复现入口", "R47包内", "Reproduction entry points", "复现入口"),
    (13, "本包", "F0_图注要点", "本轮新增", "Figure notes: interval types", "图内说明与图注必须包含的要点"),
    (13, "本包", "F1–F6 图源", "本轮新增", "Figure source data", "六张主图每个点的来源（F2–F5 与 R47 逐值相同）"),
    (13, "本包", "E1_环境依赖", "本轮新增", "Software environment", "第三方依赖与确切版本（从代码闭包扫描）"),
    (13, "本包", "E2_路径说明", "本轮新增", "Hard-coded paths and project layout", "写死的绝对路径与项目内目录（从代码闭包扫描）"),
]
NEW_SHEETS = {"O1_有限D候选硬选择oracle", "F0_图注要点", "E1_环境依赖", "E2_路径说明"}


def _sheet_name(n, sheet):
    s = f"S{n}_{sheet}"
    return s[:31]


def s_index():
    """返回 (索引表, {S 表名: 源表 DataFrame})。接回的表逐值复制；行数、交付副本 SHA 实时读取。"""
    sha_of = {k: sha16(delivered_bytes(k)) for k in INP}
    rows, copies, n = [], {}, 0
    for e in IDX:
        e = list(e) + [""] * (9 - len(e))
        g, src, sheet, how, en, cn, note, succ, succ_how = e
        r = dict(组=G[g], 建议编号="", 来源=src, 工作表=sheet, 处理=how, 英文标题=en, 内容=cn, 备注=note, 取代者=succ, 取代方式=succ_how,
                 本包表名="", 行数=np.nan, 交付副本="", 交付SHA16="")
        if how in ("R47包内", "接回", "本轮新增"):
            n += 1; r["建议编号"] = f"S{n}"
        if src in INP:
            ref = INP[src][1]
            r["交付副本"] = str(ref[1].relative_to(HAND)) + ("" if ref[0] == "file" else f" ∷ {ref[2]}")
            r["交付SHA16"] = sha_of[src]
            d = rd(src, sheet)
            r["行数"] = len(d)
            if how == "接回":
                nm = _sheet_name(n, sheet)
                assert nm not in copies, nm
                copies[nm] = d; r["本包表名"] = nm
        elif src == "本包" and sheet in NEW_SHEETS:
            r["本包表名"] = sheet
        elif src == "R47代码包":
            r["交付副本"] = str(Z47.relative_to(HAND)); r["交付SHA16"] = sha16(Z47.read_bytes())
        rows.append(r)
    return pd.DataFrame(rows), copies


def containment_checks(idx):
    """「已被取代 / 包含（逐位）」的声明必须成立：旧表每一行都能在取代表中找到逐值相同的行（共同列；列名的 . 与 _ 视为同一）。"""
    out = []
    for r in idx[idx.取代方式 == "包含（逐位）"].itertuples():
        old = rd(r.来源, r.工作表).rename(columns=lambda c: c.replace(".", "_"))
        sk, ss = r.取代者.split("∷")
        new = rd(sk, ss).rename(columns=lambda c: c.replace(".", "_"))
        if "指标" in new.columns and "指标" not in old.columns:
            new = new[new.指标.astype(str).str.lower() == "brier"]
        cols = [c for c in old.columns if c in new.columns]
        key = lambda d: d[cols].astype(str).agg("|".join, axis=1)
        miss = int((~key(old).isin(set(key(new)))).sum())
        out.append(dict(旧表=f"{r.来源}∷{r.工作表}", 取代表=r.取代者, 共同列=len(cols), 旧表行数=len(old), 找不到的行=miss, 通过=miss == 0 and len(cols) >= 4))
    return pd.DataFrame(out)


# ── E：环境依赖与路径（从代码闭包实时扫描）───────────────────────
ENTRY = ["pgss_r48_sources", "pgss_r48_figures", "pgss_r48_report", "pgss_r48_package", "pgss_r48_package_tests",
         "pgss_r47_numbers", "pgss_r47_figures", "pgss_r47_report", "pgss_r47_example", "pgss_r47_package", "pgss_r47_package_tests",
         "pgss_r383_build_v5", "pgss_xgb_run", "pgss_xgb_engine", "pgss_r42_inference",
         "pgss_r44_nhanes_cohort", "pgss_r44_nhanes_run", "pgss_r44_nhanes_inference",
         "pgss_r45_nwicu_build", "pgss_r45_nwicu_run", "pgss_r45_nwicu_inference", "pgss_r45_target", "pgss_r45_target_inference",
         "pgss_r45_plasmode", "pgss_r45_plasmode_summary", "pgss_r43_t1", "pgss_r43_t1_summary"]


def closure():
    import pgss_r46_package as P46
    return P46.closure(ENTRY + P46.ENTRY), P46.NHMOD


PIP = {"sklearn": "scikit-learn", "psycopg2": "psycopg2-binary", "PIL": "pillow"}
ROLE = {"numpy": "全部数值计算", "pandas": "表格与读写", "scipy": "推断与优化", "scikit-learn": "交叉验证工具、Platt 校准",
        "xgboost": "冻结引擎 xgb-1.0", "threadpoolctl": "线程固定为 1", "psutil": "运行资源记录", "matplotlib": "主图",
        "psycopg2-binary": "只用于 MIMIC-IV/eICU-CRD data v5 构建（PostgreSQL）", "pyarrow": "parquet 读写（pandas 引擎，未直接 import）",
        "openpyxl": "xlsx 读写（pandas 引擎，未直接 import）", "pillow": "由 PNG 转出 300 dpi TIFF（图脚本与包测试）"}
INDIRECT = ["pyarrow", "openpyxl", "pillow"]


def e1_env():
    code, nh = closure()
    found = {}
    for f in [HERE / c for c in code] + [nh]:
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for m in mods:
                found.setdefault(m.split(".")[0], set()).add(f.name)
    local = {pathlib.Path(c).stem for c in code} | {"nhanes"}
    rows = [dict(包="python", 版本=sys.version.split()[0], 直接导入的文件数=np.nan, 作用="解释器（R41–R47 均在 Windows 11 上运行）")]
    for m in sorted(k for k in found if k not in sys.stdlib_module_names and k not in local):
        p = PIP.get(m, m)
        rows.append(dict(包=p, 版本=IM.version(p), 直接导入的文件数=len(found[m]), 作用=ROLE.get(p, "")))
    for p in INDIRECT:
        if p not in {r["包"] for r in rows}:                       # 已被直接导入扫到的不再重复登记
            rows.append(dict(包=p, 版本=IM.version(p), 直接导入的文件数=0, 作用=ROLE[p]))
    return pd.DataFrame(rows)


def requirements_txt(e1):
    L = ["# PGSS — exact versions used for all frozen runs (R41–R47) and for R48 figures; Python " + e1.loc[e1.包 == "python", "版本"].iloc[0]]
    for r in e1[e1.包 != "python"].itertuples():
        L.append(f"{r.包}=={r.版本}" + ("    # only for building data v5 from PostgreSQL" if r.包 == "psycopg2-binary" else ""))
    return "\n".join(L) + "\n"


ABS_RX = re.compile(r'["\']([A-Za-z]:[/\\][^"\']*)["\']')
DIR_RX = re.compile(r'(?:ROOT|HERE\.parent)\s*/\s*["\']((?:\d{2}_|1_数据|00_PROTOCOL)[^"\']*)["\']')
ABS_ROLE = {
    "E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data": ("NWICU 0.1.0 原始数据目录（PhysioNet 受限数据）", "改为本地 NWICU 0.1.0 的 data 目录"),
    "D:/NAH公共数据库/NHANES/2_代码": ("NHANES 辅助模块 nhanes.py 所在目录", "包内 code/external/nhanes.py 即该模块；把 code/external 放进 sys.path 即可"),
    "D:/NAH公共数据库/NHANES/2_代码/nhanes.py": ("打包时取 NHANES 辅助模块的位置", "只影响打包，不影响计算"),
    "D:/Research/3_对话与交接/GPT交接_PGSS": ("交付副本目录", "只用于与交付副本逐字节比对，不参与计算；复现者可删去该核对"),
}
DIR_ROLE = {
    "19_R38": "冻结 MIMIC-IV/eICU-CRD 分析数据（data v5 在 19_R38/data_v5）", "16_R35_FIX": "R35 线性基准旧数据（只被 LR 分支与旧 QC 引用；XGB 结果不依赖）",
    "22_R41_Stage0": "R41 Stage 0 输出", "23_R42_正式主实验": "ICU 主实验逐单元产物（患者级预测只留本地）与推断",
    "24_R43_结果补齐与T1": "R43 只读补齐与 T1", "25_R44_NHANES跨领域复现": "NHANES 队列与运行", "26_R45_NWICU外部数据准备": "NWICU 准备记录",
    "27_R45_外部复现与机制核验": "NWICU 运行、NHANES 目标一致性、plasmode", "28_R46_最终核验与T2": "R46 修正重跑与 T2",
    "29_R47_写作交接": "R47 数字包、主图与 T2 记录", "30_R48_图稿与补充材料": "本轮输出", "00_PROTOCOL": "SAP 文件", "1_数据": "NHANES 本地库（nhanes.py 的 ROOT 之下）",
}


def e2_paths():
    code, nh = closure()
    rows = []
    for f in [HERE / c for c in code] + [nh]:
        src = f.read_text(encoding="utf-8").splitlines()
        tree = ast.parse("\n".join(src))
        doc_lines, own_keys = set(), set()
        if f.name == pathlib.Path(__file__).name:                 # 本文件的路径说明字典（ABS_ROLE）里的键是说明文字，不是被使用的路径
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ABS_ROLE" for t in n.targets):
                    own_keys = set(range(n.lineno, n.end_lineno + 1))
        for n in ast.walk(tree):                                  # 文档字符串里的路径不执行，单独标出
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                    and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
                doc_lines |= set(range(n.body[0].lineno, n.body[0].end_lineno + 1))
        for i, l in enumerate(src, 1):
            if i in own_keys:
                continue
            for m in ABS_RX.finditer(l):
                lit = m.group(1).replace("\\\\", "/").replace("\\", "/")
                role, how = ABS_ROLE.get(lit, ("", ""))
                if i in doc_lines or l.lstrip().startswith("#"):
                    role, how = "文档字符串或注释中的示例，不执行", "无需处理（公开发布前可清除示例路径）"
                rows.append(dict(类型="绝对路径", 文件=f.name if f != nh else "external/nhanes.py", 行=i, 路径=lit if "Users/" not in lit else "（个人用户目录示例，已隐去）",
                                 作用=role, 复现时怎么改=how))
            for m in DIR_RX.finditer(l):
                d = m.group(1).split("/")[0]
                rows.append(dict(类型="项目内目录", 文件=f.name if f != nh else "external/nhanes.py", 行=i, 路径=m.group(1), 作用=DIR_ROLE.get(d, ""),
                                 复现时怎么改="在代码目录的上一级建同名目录（项目根 = 代码目录的父目录），或改该行常量"))
    d = pd.DataFrame(rows).drop_duplicates(subset=["类型", "文件", "行", "路径"]).reset_index(drop=True)
    env = pd.DataFrame([dict(类型="环境变量", 文件="pgss_r383_build_v5.py、pgss_r35_extract.py", 行=np.nan, 路径=v, 作用=r, 复现时怎么改=h) for v, r, h in [
        ("PGHOST / PGPORT / PGUSER", "PostgreSQL 连接（默认 127.0.0.1 / 5431 / postgres）", "按本地数据库设置"),
        ("PGPASSWORD", "PostgreSQL 口令；缺失时数据构建脚本在导入时退出", "只通过环境变量传入，绝不写进文件"),
        ("REUSE_STAGING", "data v5 构建是否复用暂存表（默认 1）", "首次构建无需设置")]])
    return pd.concat([d, env], ignore_index=True)


def paths_md(e2):
    L = ["# Paths and environment (R48)", "",
         "The frozen scripts keep their original paths: changing them would change code fingerprints that were checked in earlier rounds.",
         "Project root = the parent directory of the code directory (`code/` in this archive, `99_代码/` in the project).",
         "Every line below was found by scanning the code closure; the same table is sheet `E2_路径说明` of `R48_图源与补充材料.xlsx`.", ""]
    for t in ("绝对路径", "项目内目录", "环境变量"):
        x = e2[e2.类型 == t]
        L += [f"## {t}", "", "| 文件 | 行 | 路径 | 作用 | 复现时怎么改 |", "|---|---|---|---|---|"]
        for r in x.itertuples():
            L.append(f"| {r.文件} | {'' if pd.isna(r.行) else int(r.行)} | `{r.路径}` | {r.作用} | {r.复现时怎么改} |")
        L.append("")
    L += ["## Not shipped", "", "Patient-level data (MIMIC-IV, eICU-CRD, NWICU), per-patient predictions, `runs/`, `_local_only/` and literature PDFs stay local.",
          "Real-data steps need credentialed PhysioNet access; NHANES steps need the public NHANES files and the local NHANES store used by `nhanes.py`.", ""]
    return "\n".join(L)


# ── 第 52 轮落实表 ───────────────────────────────────────────────
def g1_round52(S):
    f6a = S["F6a_T2部署两轨等权"]; idx = S["S0_补充材料索引"]
    cnt = idx.处理.value_counts()
    return pd.DataFrame([
        dict(条目="1", GPT要求="图 6 加入已有 CV_LB_M|P*（a 最终两轨等权改善；b 逐轨最终风险恶化；c 重复级至少一轨不达标启用）；只读既有冻结表；图注写明 FCT 部署 F 训练模型、b 不是 c 的错误启用率",
             做法="F6a/F6b/F6c 原样取 R47 C3、C4、C6b 中 CV_LB_M|P* 的行；图内说明与 F0 图注要点写明两句", 位置="F6a–F6c、F0、Fig6",
             结果=f"F6a 选择器 {f6a.选择器.nunique()} 个（R47 为 5 个）；F6c 选择器 {S['F6c_T2启用错误'].选择器.nunique()} 个"),
        dict(条目="2", GPT要求="图 1 收紧「selection procedures applied as in real data」；NHANES 各图 Population-weighted 统一为 Sample-proportion-weighted；图 2–5 不整体重设计、不改源数值",
             做法="T2 框副标题改为只在合成数据上运行；选择程序一行注明在哪里运行；全部图中的 population-weighted 文字改为 sample-proportion-weighted；F2–F5 数值列与 R47 逐值相同",
             位置="Fig1、Fig3、F3a",
             结果="见包测试「图文字清单」与「图源与 R47」；Fig1 L2 框的 general population 是对 NHANES 人群的描述，不是尺度名，保留"),
        dict(条目="3", GPT要求="六张图提供 PNG 和 300 dpi TIFF，PDF 保留；主图注明真实数据点态区间、max-T 同时判断和模拟均值区间的区别",
             做法="每张图出 PNG、TIFF（LZW、RGB、300 dpi）、PDF；TIFF 像素与 PNG 相同；Fig2–Fig6 图内加区间类型说明（文字取自 F0）", 位置="figures/、F0",
             结果="见包测试「TIFF 为 300 dpi 且与 PNG 像素相同」"),
        dict(条目="4", GPT要求="有限 D 候选硬选择 oracle 只作事后描述性补充；不是全部方法的上界；不重跑 T2；不改写为预定确认性结果",
             做法="从 T2 逐重复记录只读派生 O1（两个候选集），作为与 GPT 复核包逐值比对的独立复算；表头写明性质", 位置="O1、S0",
             结果="与 GPT 复核包的比对状态见报告第 4 节"),
        dict(条目="5", GPT要求="补充表索引接回既有完整对照、辅助性能和各模型自身校准结果；保留冻结核心脚本，补齐环境依赖与路径说明，不做全面重构",
             做法="S0 逐表登记来源与交付副本 SHA；接回的表复制进本包并与源表逐值比较；旧版表标「已被取代」并写明取代者；E1/E2 从代码闭包扫描；冻结脚本不改",
             位置="S0、S 表、E1、E2、requirements.txt、PATHS.md",
             结果="；".join(f"{k} {int(cnt.get(k, 0))}" for k in ("R47包内", "接回", "本轮新增", "仅索引", "已被取代", "缺口"))),
    ])


def build():
    S = {}
    S["A1_输入与交付副本比对"] = a1_inputs()
    S["A2_参照定义核对"] = a2_reference_check()
    F = f_sources()
    S["F0_图注要点"] = f0_notes()
    S.update(F)
    S["O1_有限D候选硬选择oracle"] = o1_oracle()
    idx, copies = s_index()
    S["S0_补充材料索引"] = idx
    S["A3_取代声明核对"] = containment_checks(idx)
    S["E1_环境依赖"] = e1_env()
    S["E2_路径说明"] = e2_paths()
    S["G1_第52轮落实"] = g1_round52(S)
    S["A7_区间常量"] = a7_constants()
    S["_copies"] = copies
    return S


if __name__ == "__main__":
    S = build()
    for k, v in S.items():
        if isinstance(v, pd.DataFrame):
            print(k, v.shape)
    print(S["A1_输入与交付副本比对"].to_string())
    print(S["A2_参照定义核对"].to_string())
    print(S["A3_取代声明核对"].to_string())
    print(S["S0_补充材料索引"].处理.value_counts().to_string())
    print(len(S["_copies"]), "张 S 表")
