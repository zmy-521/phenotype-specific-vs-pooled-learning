# -*- coding: utf-8 -*-
r"""R47 写作交接 · 冻结数字包（GPT 第 51 轮 §8）。只读已交付的结果表；不拟合、不模拟、不改任何冻结目录。

输入：R41–R46 已交付给 GPT 的结果表。A2 把每个输入与交接目录中的交付副本（或交付 zip 内的副本）逐字节比对。
输出：build() 返回 {表名: DataFrame}，由 pgss_r47_report.py 与 md 同源写出 29_R47_写作交接/R47_冻结数字包.xlsx；
      图源表（F1–F6）同时被 pgss_r47_figures.py 读取作图，图中每个点都来自这些表。
报告层更正（GPT 第 51 轮 §7、§8.2）全部落在本脚本：
  ① T1 与 T2 用数值逐项并列（C7），不用「Q1–Q5 推翻 0」概括；
  ② 负的点估计与区间支持的负向结果分开（「均值±1.96SE方向」「95%区间方向」列）；
  ③ 极端值与尾部比例改标「初始 10 万评价样本」（C9；依据 pgss_r46_t2.py 中 ext 只取第 0 段 Ev0）；
  ④ 两两排序反转标为未作多重调整的描述性比较（C8）；
  ⑤ FCT 的启用错误标为指定生成条件下的 Monte Carlo 估计，不是错误率保证（C6）；
  ⑥ MIX 的平均改善、逐轨实际变差与重复间变异并列（C5）。
「主张」一栏取自 GPT 第 51 轮原文的最窄表述并注明出处，Claude 不另写带方向的结论。
"""
import sys, json, pathlib, hashlib, zipfile, inspect, re
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
HAND = pathlib.Path(r"D:/Research/3_对话与交接/GPT交接_PGSS")
OUT = ROOT / "29_R47_写作交接"
R46 = ROOT / "28_R46_最终核验与T2"
Z46 = HAND / "46" / "R46_CODE_TESTS.zip"
# 输入登记：名称 → (本地路径, 交付副本)；交付副本 = ("file", 路径) 或 ("zip", zip 路径, 成员名)
INP = {
    "R46证据表": (R46 / "R46_最终核验与T2_证据表.xlsx", ("file", HAND / "46" / "R46_最终核验与T2_证据表.xlsx")),
    "R41证据表": (ROOT / "22_R41_Stage0" / "R41_Stage0_证据表.xlsx", ("file", HAND / "41" / "R41_Stage0_证据表.xlsx")),
    "R42推断": (ROOT / "23_R42_正式主实验" / "inference" / "R42_INFERENCE.xlsx",
              ("zip", HAND / "42" / "R42_CODE_TESTS.zip", "inference/R42_INFERENCE.xlsx")),
    "R43只读": (ROOT / "24_R43_结果补齐与T1" / "readonly" / "R43_READONLY.xlsx",
              ("zip", HAND / "43" / "R43_CODE_TESTS.zip", "readonly/R43_READONLY.xlsx")),
    "NWICU推断v2": (R46 / "nwicu_v2" / "inference" / "R45_NWICU_INFERENCE.xlsx",
                  ("zip", Z46, "r46/nwicu_v2/inference/R45_NWICU_INFERENCE.xlsx")),
    "NHANES推断v2": (R46 / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx",
                   ("zip", Z46, "r46/nhanes_v2/r44v2/inference/R44_INFERENCE.xlsx")),
    "NHANES目标一致性v2": (R46 / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx",
                       ("zip", Z46, "r46/nhanes_v2/target_v2/R45_TARGET_INFERENCE.xlsx")),
    "plasmode汇总": (ROOT / "27_R45_外部复现与机制核验" / "plasmode" / "R45_PLASMODE_SUMMARY.xlsx",
                   ("zip", HAND / "45" / "R45_CODE_TESTS.zip", "r45/plasmode/R45_PLASMODE_SUMMARY.xlsx")),
    "文献表": (R46 / "lit" / "R46_LIT_TABLE.xlsx", ("zip", Z46, "r46/lit/R46_LIT_TABLE.xlsx")),
}
CONDS = ["N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H"]
SELS = ["ALWAYS_SPLIT", "CV_POINT_M|P*", "FCT|P*", "MIX", "POOLMIX"]          # 主图选择器（ALWAYS_POOLED = 参照，恒为 0）
M = 0.01
_X = {}
TRACE = []            # 本模块格式化进文字的每个原值（报告脚本据此做第二层逐值追踪）


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def rd(name, sheet):
    k = (name, sheet)
    if k not in _X:
        _X[k] = pd.read_excel(INP[name][0], sheet_name=sheet)
    return _X[k].copy()


def ev(sheet):
    return rd("R46证据表", sheet)


# ── 方向分类 ─────────────────────────────────────────────────────
def side_t2(mean, se):
    """T2：重复间均值 ± 1.96·SE 相对 0 的位置。"""
    if se == 0 and mean == 0:
        return "全部重复为 0"
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return "区间整体 > 0" if lo > 0 else "区间整体 < 0" if hi < 0 else "区间跨 0"


def side_m(mean, se, m=M):
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return "区间整体 ≥ m" if lo >= m else "区间整体 < m" if hi < m else "区间跨 m"


def add_ci(d, mean, se, pre=""):
    d[f"{pre}下"] = d[mean] - 1.96 * d[se]
    d[f"{pre}上"] = d[mean] + 1.96 * d[se]
    d[f"{pre}均值±1.96SE方向"] = [side_t2(a, b) for a, b in zip(d[mean], d[se])]
    return d


# ── A：更正与输入 ─────────────────────────────────────────────────
def a2_inputs():
    rows = []
    for nm, (p, ref) in INP.items():
        b = p.read_bytes()
        if ref[0] == "file":
            rb = ref[1].read_bytes(); where = str(ref[1].relative_to(HAND))
        else:
            rb = zipfile.ZipFile(ref[1]).read(ref[2]); where = f"{ref[1].relative_to(HAND)} ∷ {ref[2]}"
        rows.append(dict(输入=nm, 本地路径=str(p.relative_to(ROOT)), 本地SHA16=sha16(b), 交付副本=where, 交付SHA16=sha16(rb),
                         逐字节相同=b == rb))
    return pd.DataFrame(rows)


# ── B：真实数据三层 ──────────────────────────────────────────────
PH_ORDER = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]


def b1_tasks():
    a = rd("R41证据表", "A1_任务与轨")
    r = []
    for x in a.itertuples():
        r.append(dict(库=x.库, 结局=x.结局, 表型=x.表型, 纳入=True, n=x.n, 事件=x.事件, A轨n=x.全体A_n, A轨事件=x.全体A_事件,
                      B轨n=x.全体B_n, B轨事件=x.全体B_事件,
                      分组单位="患者" if x.库 == "mimic" else "任务内患者—医院连通分量", 来源="R41 A1_任务与轨（data v5）"))
    nw = rd("NWICU推断v2", "1_任务纳入")
    for x in nw.itertuples():
        r.append(dict(库="nwicu", 结局=x.结局, 表型=x.表型, 纳入=bool(x.纳入), n=x.任务人次, 事件=x.事件, A轨n=x.A轨人次, A轨事件=x.A轨事件,
                      B轨n=x.任务人次 - x.A轨人次, B轨事件=x.事件 - x.A轨事件, 分组单位="患者",
                      来源="NWICU v2 推断 1_任务纳入（住院死亡已排除无生命状态证据的 200 人次）"))
    nh = rd("NHANES推断v2", "1_任务纳入")
    for x in nh.itertuples():
        r.append(dict(库="nhanes", 结局=x.结局, 表型=x.表型, 纳入=bool(x.纳入), n=x.任务人数, 事件=x.事件, A轨n=x.A轨人数, A轨事件=x.A轨事件,
                      B轨n=x.任务人数 - x.A轨人数, B轨事件=x.事件 - x.A轨事件, 分组单位="个体（SEQN）",
                      来源="NHANES v2 推断 1_任务纳入" + ("" if x.纳入 else "（未纳入：外层折内早停留出单类别，不可执行）")))
    d = pd.DataFrame(r)
    d["事件率"] = d.事件 / d.n; d["A轨占比"] = d.A轨n / d.n
    d["A轨事件率"] = d.A轨事件 / d.A轨n; d["B轨事件率"] = d.B轨事件 / d.B轨n
    return d[["库", "结局", "表型", "纳入", "n", "事件", "事件率", "A轨n", "A轨占比", "A轨事件", "A轨事件率", "B轨n", "B轨事件", "B轨事件率",
              "分组单位", "来源"]]


def b5_nhanes_case():
    t = rd("NHANES目标一致性v2", "2_Brier对照")
    t = t[(t.结局 == "O_MORT10") & (t.表型 == "低白蛋白")].copy()
    MEAN = {"E1": "原预定比较：HTFS vs P*（P* 按冻结引擎的两轨等权目标开发）",
            "C1": "目标一致（人口加权）：HTFS_pop vs P*_pop",
            "C2": "合并参照改目标（人口加权）：P*_pop vs P*", "C3": "HTFS vs P*_pop",
            "D1": "目标一致（B 轨）：HTFS_B vs P*_B", "D2": "合并参照改目标（B 轨）：P*_B vs P*", "D3": "HTFS vs P*_B"}
    LAY = {"E1": "L2 原始跨领域复现（预定）", "C1": "L3 结果知情后追加对照（运行前登记）", "D1": "L3 结果知情后追加对照（运行前登记）"}
    t["对照含义"] = t.对照.map(MEAN); t["证据层"] = t.对照.map(LAY).fillna("L3 追加对照的分解比较（描述）")
    t["与目标一致的尺度"] = ((t.对照 == "E1") & (t.尺度 == "bal")) | (t.对照.isin(["C1", "C2", "C3"]) & (t.尺度 == "pop")) | \
        (t.对照.isin(["D1", "D2", "D3"]) & (t.尺度 == "B"))
    t = t.rename(columns={"三态_m0.010": "三态_m1", "相对0方向": "95%区间方向"})
    return t[["证据层", "对照", "对照含义", "参照", "候选", "尺度", "与目标一致的尺度", "R参照", "观察绝对差", "观察相对改善", "相对_P2_5", "相对_P97_5",
              "三态_m1", "95%区间方向"]].reset_index(drop=True)


def b6_gap():
    g = rd("NHANES目标一致性v2", "6_差距收回")
    g = g[(g.结局 == "O_MORT10") & (g.表型 == "低白蛋白")].copy()
    g = g.rename(columns={"R_Pstar": "R_P*（两轨等权开发）", "R_Pstar_q": "R_P*_目标", "R_HTFS_q": "R_HTFS_目标", "R_HTFS": "R_HTFS（原）",
                          "v10差距": "原差距（R_P* − R_HTFS）", "收回": "收回（R_P* − R_P*_目标）", "收回比例_相对v10差距": "收回比例"})
    return g.reset_index(drop=True)


# ── C：T2 与模拟 ─────────────────────────────────────────────────
C_MOD = {"N1": "T2-A", "N2": "T2-A", "P1": "T2-A", "P2": "T2-A", "N1-T": "T2-A", "P2-T": "T2-A", "N1-H": "T2-B", "P2-H": "T2-B"}
C_X = {"T2-A": "高斯 N(0, Σ)，Σ = 0.2·11ᵀ + 0.8·I（23 维）", "T2-B": "多元 t₅：X = √(3/5)·G/√(V/5)，G ~ N(0, Σ)，V ~ χ²₅"}
C_F = {"N1": "无轨道差异：logit p = −1.6 + 共同项", "N2": "只有截距差异：−1.9 + 共同项 + 0.8·Z",
       "P1": "表型相关交互：−1.6 + 共同项 + 1.0·Z·X₆", "P2": "轨道特异非线性稀疏结构：A 轨 −1.4 + g_A(X)，B 轨 −1.7 + g_B(X)"}


def c1_design():
    d1 = ev("D1_T2生成与评价规模"); d10 = ev("D10_T2成本与失败")
    keep = ["条件", "重复", "开发事件率均值", "评价A轨平均p", "评价B轨平均p", "A轨Bayes风险", "B轨Bayes风险", "A轨开发人数均值", "A轨开发事件均值",
            "B轨开发事件均值", "评价10万", "评价50万", "评价100万", "最终仍有近边界项的重复"]
    d = d1[keep].merge(d10[["条件", "每重复秒均值", "拟合次数均值", "峰值内存MB最大"]], on="条件")
    d.insert(1, "模块", d.条件.map(C_MOD))
    d.insert(2, "协变量", d.模块.map(C_X))
    d.insert(3, "Z", ["1[X₈ > Φ⁻¹(0.85)]（阈值定义）" if c.endswith("-T") else "Bernoulli(0.15)，与 X 独立" for c in d.条件])
    d.insert(4, "结局函数", [C_F[c.split("-")[0]] for c in d.条件])
    d.insert(5, "开发集n", 6000); d.insert(6, "π_A", 0.15)
    s = ev("A3_运行完成情况"); t2 = s[s.部分.str.startswith("T2")].iloc[0]
    d["失败"] = 0 if int(t2.不可执行) == 0 else np.nan
    d = d.rename(columns={"重复": "完成重复"})
    return d


def c2_pairs():
    d = ev("D2_T2候选对真值收益")
    d = add_ci(d, "真值相对收益均值", "SE")
    d["相对m方向"] = [side_m(a, b) for a, b in zip(d.真值相对收益均值, d.SE)]
    cols = ["条件", "候选对", "轨", "重复", "真值相对收益均值", "SE", "下", "上", "均值±1.96SE方向", "相对m方向", "P5", "中位", "P95",
            "达m_确定_分子", "达m_确定_分母", "为负_确定_分子", "为负_确定_分母", "m边界不确定", "零边界不确定"]
    return d[cols]


def c3_bal():
    d = ev("X4_F6_选择错误与部署风险_两轨等权")          # 全部 13 个选择器（D3 是其中 10 个）
    d = add_ci(d, "均值", "SE")
    return d[["条件", "选择器", "重复", "均值", "SE", "下", "上", "均值±1.96SE方向", "P5", "中位", "P95", "为负重复"]].rename(
        columns={"均值": "两轨等权部署改善均值", "为负重复": "点估计为负的重复"})


def c4_track():
    d = ev("X4_F6_选择错误与部署风险_逐轨")              # 全部 13 个选择器（D4 是其中 10 个）
    d = add_ci(d, "部署相对PstarD改善均值", "SE")
    cols = ["条件", "选择器", "轨", "重复", "部署相对PstarD改善均值", "SE", "下", "上", "均值±1.96SE方向", "P5", "中位", "P95",
            "部署实际变差_分子", "部署实际变差_分母", "部署实际变差_CP下", "部署实际变差_CP上", "部署变差边界不确定"]
    return d[cols]


def c5_mix(C3, C4):
    rows = []
    for s in ("MIX", "POOLMIX", "ALWAYS_SPLIT", "CV_POINT_M|P*", "FCT|P*"):
        for c in CONDS:
            b = C3[(C3.条件 == c) & (C3.选择器 == s)]
            if b.empty:
                continue
            b = b.iloc[0]; r = dict(选择器=s, 条件=c, 两轨等权均值=b.两轨等权部署改善均值, 两轨等权SE=b.SE, 两轨等权方向=b["均值±1.96SE方向"],
                                    两轨等权P5=b.P5, 两轨等权P95=b.P95, 两轨等权点估计为负=b.点估计为负的重复)
            for g in ("A", "B"):
                t = C4[(C4.条件 == c) & (C4.选择器 == s) & (C4.轨 == g)].iloc[0]
                r.update({f"{g}轨均值": t.部署相对PstarD改善均值, f"{g}轨SE": t.SE, f"{g}轨方向": t["均值±1.96SE方向"], f"{g}轨P5": t.P5,
                          f"{g}轨P95": t.P95, f"{g}轨实际变差": int(t.部署实际变差_分子), f"{g}轨重复": int(t.部署实际变差_分母),
                          f"{g}轨实际变差_CP上": t.部署实际变差_CP上})
            rows.append(r)
    return pd.DataFrame(rows)


def c6_fct():
    d4 = ev("X4_F6_选择错误与部署风险_逐轨"); d5 = ev("D5_T2重复级至少一轨"); d7 = ev("D7_T2认证样本成本")
    sel = ["FCT|P*", "FCT|S0*", "USTUN_ADAPT", "CV_POINT_M|P*", "CV_LB_M|P*"]
    a = d4[d4.选择器.isin(sel)][["条件", "选择器", "轨", "启用_分子", "启用_分母", "不达标启用_条件于确定不达标_分子", "不达标启用_条件于确定不达标_分母",
                               "不达标启用_条件于确定不达标_CP上", "检出_条件于确定达标_分子", "检出_条件于确定达标_分母", "检出_条件于确定达标_CP下",
                               "检出_条件于确定达标_CP上", "启用_m边界不确定"]].copy()
    a.insert(0, "证据性质", "指定生成条件下 200 次重复的 Monte Carlo 估计；不是错误率保证")
    b = d5[d5.选择器.isin(sel)][["条件", "选择器", "独立重复", "≥1轨不达标启用_分子", "≥1轨不达标启用_分母", "≥1轨不达标启用_CP下",
                               "≥1轨不达标启用_CP上", "≥1轨实际变差启用_分子", "≥1轨实际变差启用_CP上"]].copy()
    b.insert(0, "证据性质", "指定生成条件下 200 次重复的 Monte Carlo 估计；不是错误率保证")
    c = add_ci(d7.copy(), "P星F相对P星D改善均值", "SE")
    c = c.rename(columns={"P星F相对P星D改善均值": "P*_F相对P*_D改善均值"})
    c.insert(0, "含义", "认证留出 1/3 后，只用 2/3 开发数据拟合的合并参照相对用全部开发数据者的真值改善（样本成本）")
    return a.reset_index(drop=True), b.reset_index(drop=True), c


def c7_t1t2():
    d = ev("D8_T2与T1并列")
    d = add_ci(d, "T2均值", "T2_SE", pre="T2")
    d = d.drop(columns=["T2下", "T2上"]).rename(columns={"T2均值±1.96SE方向": "T2方向"})
    d["T2下"] = d.T2均值 - 1.96 * d.T2_SE; d["T2上"] = d.T2均值 + 1.96 * d.T2_SE
    s1 = np.sign(d.T1均值)
    cls = []
    for t1, sd in zip(s1, d.T2方向):
        if sd == "全部重复为 0":
            cls.append("T2 全部重复为 0")
        elif sd == "区间跨 0":
            cls.append("T2 区间跨 0")
        else:
            s2 = 1 if sd == "区间整体 > 0" else -1
            cls.append("T2 区间支持的方向与 T1 均值同号" if s2 == t1 else "T2 区间支持的方向与 T1 均值异号" if t1 != 0 else "T1 均值为 0")
    d["T1与T2"] = cls
    d.insert(0, "地位", "T1 = 设计阶段探索（每条件 20 次）；T2 = 采用（每条件 200 次）")
    return d[["地位", "项", "条件", "对象", "轨", "T1重复", "T1均值", "T1_SE", "T2重复", "T2均值", "T2_SE", "T2下", "T2上", "T2方向", "T1与T2"]]


def c8_rank():
    d = ev("D9_T2排序推翻"); x = ev("X5_T2两两排序全表")
    n_cmp = int(x.T1均值差.notna().sum()); n_cond = int(x[x.T1均值差.notna()].条件.nunique())
    d.insert(0, "性质", f"描述性：有 T1 对应的 {n_cond} 个条件共 {n_cmp} 个选择器对逐一比较，未作多重调整；不是确认性发现")
    return d, n_cmp


def c9_extreme():
    d = ev("D1_T2生成与评价规模")
    ext = ["评价X绝对值最大_中位", "评价X绝对值最大_最大", "评价X绝对值P99_9_均值", "评价X绝对值超4比例_均值", "线性预测子P0_1_均值", "线性预测子P99_9_均值",
           "线性预测子最小", "线性预测子最大", "p小于1e6比例_均值", "p大于1减1e6比例_均值"]
    d = d[["条件", "重复"] + ext].rename(columns={c: "初始10万评价样本_" + c.replace("评价", "") for c in ext})
    src = (HERE / "pgss_r46_t2.py").read_text(encoding="utf-8")
    line = next((l.strip() for l in src.splitlines() if l.strip().startswith("ext = dict(") and "Ev0" in l), "")
    d.insert(2, "评价范围", "每次重复最初的 100,000 个评价样本（第 0 段），不是追加后的 50 万或 100 万")
    return d, line


def c10_plasmode():
    b = rd("plasmode汇总", "B_候选对真值收益"); dd = rd("plasmode汇总", "D_选择器两轨等权")
    b = b[b.候选对 == "D|P*"][["δ", "候选对", "轨", "重复", "真值相对收益均值", "SE"]].assign(表="候选对真值收益（HTFS_D vs P*_D）")
    dd = dd[dd.选择器.isin(["MIX", "FCT|P*", "ALWAYS_SPLIT", "CV_POINT_M|P*"])][["δ", "选择器", "重复", "均值", "SE"]].assign(表="选择器两轨等权部署改善")
    d = pd.concat([b.rename(columns={"真值相对收益均值": "均值"}), dd], ignore_index=True)
    d.insert(0, "地位", "探索性（R45 plasmode：真实 MIMIC 协变量结构 + 指定生成机制；每 δ 20 次）")
    return d[["地位", "表", "δ", "候选对", "选择器", "轨", "重复", "均值", "SE"]]


# ── D：写作资产 ──────────────────────────────────────────────────
def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    h = df[m]
    assert len(h) == 1, (kw, len(h))
    return h.iloc[0]


def p2(v):
    TRACE.append(float(v))
    return f"{100 * v:.2f}%"


def d1_matrix(S):
    """主张取自 GPT 第 51 轮原文（注明节号）的最窄表述；关键数字逐个从本包的表取出。"""
    B3, B5, C2, C3, C4, C5, C6b, C6c, C7, C8 = (S[k] for k in ("B3_分层计数", "B5_NHANES低白蛋白归因", "C2_T2候选对真值收益", "C3_T2两轨等权部署",
                                                              "C4_T2逐轨部署与变差", "C5_MIX等三面并列", "C6b_T2重复级启用错误",
                                                              "C6c_T2认证样本成本", "C7_T1与T2数值并列", "C8_两两排序反转_描述"))
    R = []

    def add(**k):
        R.append(k)
    icu = B3[B3.层.str.startswith("L1") & (B3.尺度 == "bal")]
    add(编号="K1", 出处="GPT 第51轮 §三.1", 证据层级="预定主要比较（MIMIC、eICU）+ 同一程序的独立复现（NWICU）",
        主张="三个 ICU 数据库的 36 个预定任务中，HTFS 相对 P* 的两轨等权 Brier 改善没有任务被区间支持达到 m",
        估计对象="E1 两轨等权相对 Brier 改善（固定 OOF 预测，联合簇重抽 95% 区间）",
        关键数字="；".join(f"{r.库} {int(r.任务)} 任务：排除 {int(r.排除)}、证据不足 {int(r.证据不足)}、支持 {int(r.支持)}，区间整体 < 0 {int(r.全负)}"
                       for r in icu.itertuples()),
        证据表="B3_分层计数（L1、bal）；B2_三层总结果表",
        限制="36 个任务共享患者、表型相互重叠，不是 36 项独立试验；区间不含算法级重训变异；eICU 有效独立簇少；NWICU 住院死亡结局在看到 R45 结果后修正",
        不应写成="临床表型无价值；所有疾病都不应分轨；47 项独立试验全部失败")
    nh = B3[B3.层.str.startswith("L2")]
    nb, nB, nP = (pick(nh, 尺度=s) for s in ("bal", "B", "pop"))
    add(编号="K2", 出处="GPT 第51轮 §三.1", 证据层级="跨领域复现（NHANES，一般人群长期死亡）",
        主张="NHANES 原始比较在主要尺度（两轨等权）上没有任务被支持；保留证据不足与逐轨不一致的情况",
        估计对象="E1（同 K1）", 关键数字=f"bal {int(nb.任务)} 任务：排除 {int(nb.排除)}、证据不足 {int(nb.证据不足)}、支持 {int(nb.支持)}；"
                                  f"B 轨支持 {int(nB.支持)}；人口加权支持 {int(nP.支持)}",
        证据表="B3_分层计数（L2）", 限制="不是未触碰数据；不使用调查权重；O_CVD10×低白蛋白不可执行",
        不应写成="NHANES 证实专业化无效")
    e1b = pick(B5, 对照="E1", 尺度="B"); d1b = pick(B5, 对照="D1", 尺度="B"); e1p = pick(B5, 对照="E1", 尺度="pop"); c1p = pick(B5, 对照="C1", 尺度="pop")
    add(编号="K3", 出处="GPT 第51轮 §三.2", 证据层级="结果知情后追加对照（运行前登记），单一任务",
        主张="NHANES 低白蛋白任务中，专属模型在原比较里的 B 轨优势，在合并程序也按 B 轨目标开发后没有保留下来；表现更好与独立训练不可替代不是同一结论",
        估计对象="E1 B 轨（HTFS vs P*）→ D1 B 轨（HTFS_B vs P*_B）；E1 人口加权 → C1 人口加权",
        关键数字=f"E1 B {p2(e1b.观察相对改善)}（{p2(e1b.相对_P2_5)}～{p2(e1b.相对_P97_5)}）→ D1 B {p2(d1b.观察相对改善)}"
                 f"（{p2(d1b.相对_P2_5)}～{p2(d1b.相对_P97_5)}，{d1b.三态_m1}）；E1 pop {p2(e1p.观察相对改善)} → C1 pop {p2(c1p.观察相对改善)}"
                 f"（{p2(c1p.相对_P2_5)}～{p2(c1p.相对_P97_5)}）",
        证据表="B5_NHANES低白蛋白归因；B6_NHANES差距收回",
        限制="在 R44 结果之后登记；只有一个任务承担预定解释性判读；不证明差异为零；不分解早停、参数、面板与校准各自的作用",
        不应写成="原阳性是假的；目标不一致是既往所有专业化收益的主要原因")
    for c in ("P2", "P2-T", "P2-H"):
        a, b = pick(C2, 条件=c, 候选对="D|P*", 轨="A"), pick(C2, 条件=c, 候选对="D|P*", 轨="B")
        s = pick(C3, 条件=c, 选择器="ALWAYS_SPLIT")
        add(编号=f"K4-{c}", 出处="GPT 第51轮 §三.3", 证据层级="独立 T2" + ("（分布压力条件）" if c.endswith("-H") else ""),
            主张=f"{c}（轨道特异结构）：两轨专属模型相对强合并参照均有真值收益",
            估计对象="HTFS_D vs P*_D 条件 Brier 真值相对收益（200 次重复均值 ± 1.96·SE）；ALWAYS_SPLIT 两轨等权部署改善",
            关键数字=f"A {p2(a.真值相对收益均值)}（{a.相对m方向}）；B {p2(b.真值相对收益均值)}（{b.相对m方向}）；两轨全部专业化 {p2(s.两轨等权部署改善均值)}",
            证据表="C2_T2候选对真值收益；C3_T2两轨等权部署", 限制="只对所设生成机制、n = 6,000、π_A = 0.15 成立；不模拟医院残差相关",
            不应写成="真实临床任务存在这种结构")
    a, b = pick(C2, 条件="P1", 候选对="D|P*", 轨="A"), pick(C2, 条件="P1", 候选对="D|P*", 轨="B")
    s = pick(C3, 条件="P1", 选择器="ALWAYS_SPLIT")
    add(编号="K5", 出处="GPT 第51轮 §三.3", 证据层级="独立 T2",
        主张="P1（表型相关交互）：一轨可能获益、另一轨受损，完整双轨的平均效果为负",
        估计对象="同 K4", 关键数字=f"A {p2(a.真值相对收益均值)}（{a['均值±1.96SE方向']}）；B {p2(b.真值相对收益均值)}（{b['均值±1.96SE方向']}，{b.相对m方向}）；"
                             f"两轨全部专业化 {p2(s.两轨等权部署改善均值)}（{s['均值±1.96SE方向']}）",
        证据表="C2、C3", 限制="同 K4", 不应写成="交互结构下总是只该专业化一轨")
    t = pick(C4, 条件="P2", 选择器="ALWAYS_SPLIT", 轨="A")
    add(编号="K6", 出处="GPT 第51轮 §三.3", 证据层级="独立 T2",
        主张="P2 中 A 轨平均改善为正，但重新开发时并非每次都改善",
        估计对象="ALWAYS_SPLIT A 轨部署相对 P*_D 的改善；实际变差 = 改善 < 0 且 |改善| ≥ 2 个积分 SE",
        关键数字=f"A 轨均值 {p2(t.部署相对PstarD改善均值)}（{t['均值±1.96SE方向']}）；实际变差 {int(t.部署实际变差_分子)}/{int(t.部署实际变差_分母)}",
        证据表="C4_T2逐轨部署与变差", 限制="同 K4", 不应写成="平均有益即每次开发都有益")
    nulls = ["N1", "N2", "N1-T", "N1-H"]
    fr = [pick(C6b, 条件=c, 选择器="FCT|P*") for c in nulls]
    add(编号="K7", 出处="GPT 第51轮 §三.4、§七.4", 证据层级="独立 T2（Monte Carlo 估计）",
        主张="在这些指定条件下，观察到 FCT|P* 较低的不达标启用比例",
        估计对象="重复级至少一轨不达标启用（无轨道特异结构的条件）",
        关键数字="；".join(f"{c} {int(r['≥1轨不达标启用_分子'])}/{int(r['≥1轨不达标启用_分母'])}（CP 上界 {p2(r['≥1轨不达标启用_CP上'])}）"
                       for c, r in zip(nulls, fr)),
        证据表="C6b_T2重复级启用错误", 限制="200 次重复的经验比例与其区间；只对所设生成机制成立",
        不应写成="已获得对所有数据分布或临床任务的错误率保证")
    f = C3[C3.选择器 == "FCT|P*"].set_index("条件").loc[CONDS]
    neg = f[f["均值±1.96SE方向"] == "区间整体 < 0"]; cr = f[f["均值±1.96SE方向"] == "区间跨 0"]
    add(编号="K8", 出处="GPT 第51轮 §三.4、§七.2", 证据层级="独立 T2",
        主张="FCT|P* 部署后的两轨等权风险，相对使用全部开发数据的合并参照 P*_D 不占优；需与相对同一拟合集参照的比较分开",
        估计对象="FCT|P* 两轨等权部署相对 P*_D 的改善；认证样本成本 = P*_F 相对 P*_D",
        关键数字=f"8 个条件点估计均 < 0；区间整体 < 0 {len(neg)}/8；区间跨 0：" +
                 "、".join(f"{c} {p2(r.两轨等权部署改善均值)}（{p2(r.下)}～{p2(r.上)}）" for c, r in cr.iterrows()) +
                 f"；样本成本 P*_F 相对 P*_D 逐轨均值 {p2(C6c['P*_F相对P*_D改善均值'].min())} 至 {p2(C6c['P*_F相对P*_D改善均值'].max())}",
        证据表="C3_T2两轨等权部署；C6c_T2认证样本成本",
        限制="认证后不重拟合是冻结设计（f = 1/3）；认证后用全部数据重拟合是未检验的变体",
        不应写成="FCT 在八个条件都已证明更差；FCT 安全且有效")
    m1 = pick(C5, 选择器="MIX", 条件="N1")
    add(编号="K9", 出处="GPT 第51轮 §三.4", 证据层级="独立 T2",
        主张="MIX 不能被宣布为普遍安全的赢家：无轨道特异结构时，两轨等权平均略正，同时 A 轨有可辨认的风险恶化",
        估计对象="MIX 两轨等权与逐轨部署改善；逐轨实际变差次数",
        关键数字=f"N1：两轨等权 {p2(m1.两轨等权均值)}（{m1.两轨等权方向}）；A 轨 {p2(m1.A轨均值)}（{m1.A轨方向}），实际变差 {m1.A轨实际变差}/{m1.A轨重复}；"
                 f"B 轨 {p2(m1.B轨均值)}，实际变差 {m1.B轨实际变差}/{m1.B轨重复}",
        证据表="C5_MIX等三面并列", 限制="预测风险恶化不等于已观察到临床伤害；只对所设机制成立", 不应写成="MIX 普遍最优或普遍安全")
    pm = S["C5b_MIX与POOLMIX配对差"]
    n2 = pick(pm, 条件="N2")
    add(编号="K10", 出处="R46 表 D6（GPT 第51轮未单独讨论）", 证据层级="独立 T2",
        主张="N2（只有截距差异）中，在所检验的融合协议下，合并候选融合 POOLMIX 的两轨等权部署风险低于 MIX",
        估计对象="两轨等权部署改善的同重复配对差（MIX − POOLMIX）",
        关键数字=f"{p2(n2.风险变化均值)}（SE {p2(n2.SE)}），MIX 更好 {int(n2.MIX更好重复)}/{int(n2.重复)}",
        证据表="C5b_MIX与POOLMIX配对差", 限制="POOLMIX 只是 P* 与另一合并族候选的一种融合", 不应写成="合并集成总优于引入专属预测")
    cg = C7[(C7.T1与T2 == "T2 区间支持的方向与 T1 均值异号")]
    add(编号="K11", 出处="GPT 第51轮 §七.1", 证据层级="T1（探索，20 次）与 T2（采用，200 次）并列",
        主张="T2 改变了部分 T1 数值的方向；正文采用 T2，T1 保留探索身份",
        估计对象="同一条件、同一对象的 T1 与 T2 均值",
        关键数字=f"T2 区间支持的方向与 T1 均值异号 {len(cg)} 行：" + "；".join(
            f"{r.条件} {r.对象} {r.轨} T1 {p2(r.T1均值)} → T2 {p2(r.T2均值)}" for r in cg.itertuples()),
        证据表="C7_T1与T2数值并列", 限制="T1 每条件 20 次", 不应写成="T2 全面确认 T1")
    n_rev = len(C8)
    add(编号="K12", 出处="GPT 第51轮 §七.1", 证据层级="独立 T2（描述性）",
        主张="选择器两两排序的 T1/T2 反转只作描述",
        估计对象="同重复配对差（均值 ± 1.96·SE）与 T1 均值差的符号", 关键数字=f"{n_rev}/{S['_n_cmp']} 个选择器对",
        证据表="C8_两两排序反转_描述", 限制="未作多重调整", 不应写成=f"{n_rev} 个确认性发现")
    add(编号="K13", 出处="GPT 第51轮 §四", 证据层级="设计事实",
        主张="T2 确认的是指定条件下的工作特性，没有识别适用于所有研究的样本量交叉点",
        估计对象="—", 关键数字="T2 开发规模固定 6,000、A 轨比例固定 0.15；8 个条件 × 200 次",
        证据表="C1_T2条件与设计", 限制="换疾病、信息集、学习器、样本结构或预测目标须重新评价", 不应写成="跨疾病自动选型的通用判据")
    return pd.DataFrame(R)[["编号", "出处", "证据层级", "主张", "估计对象", "关键数字", "证据表", "限制", "不应写成"]]


def d2_questions(S):
    """GPT 第 51 轮 §四 的评价顺序；「本研究中的估计量与证据」逐行指向本包的表。"""
    K = S["D1_主张证据限制矩阵"].set_index("编号")
    rows = [
        dict(顺序=1, 必须回答的问题="合并模型是否获得了公平的比较机会？", 对应做法="同样的可得信息、明确的开发目标、足够强的候选与合理预算",
             不应混淆="表型信息有用，不等于必须独立训练",
             本研究中的估计量="P* 的候选含显式表型（SZ、SZ-GFS、SZ-union）与轨道均衡权重；NHANES 追加对照把合并程序的开发目标改为评价尺度（C1、D1）",
             证据="K3：" + K.loc["K3", "关键数字"], 证据性质="一个结果知情后登记的真实数据任务；作为比较原则的示例，不是对既往研究的普遍解释"),
        dict(顺序=2, 必须回答的问题="独立训练是否还有剩余增量？", 对应做法="在目标一致的参照之上，进行配对的整体与逐轨评价",
             不应混淆="超过较弱参照，不等于独立训练不可替代",
             本研究中的估计量="E1：HTFS vs P*，逐轨、两轨等权、人口加权；逐轨 max-T 同时带",
             证据="K1：" + K.loc["K1", "关键数字"], 证据性质="预定主要比较 + 独立复现；固定 OOF 区间"),
        dict(顺序=3, 必须回答的问题="选择程序是否值得付出代价？", 对应做法="评价完整开发流程，并与使用全部开发数据的替代方案比较",
             不应混淆="错误启用少，不等于最终预测更好",
             本研究中的估计量="R(P*_D) − R(认证后程序) 与 R(P*_F) − R(认证后程序) 分开；样本成本 = P*_F 相对 P*_D",
             证据="K7：" + K.loc["K7", "关键数字"] + "。K8：" + K.loc["K8", "关键数字"], 证据性质="独立 T2，指定条件"),
        dict(顺序=4, 必须回答的问题="收益是否以某轨风险恶化为代价？", 对应做法="同时报告平均收益、逐轨风险及重新开发的变异",
             不应混淆="平均有益，不等于每轨、每次开发都受益",
             本研究中的估计量="逐轨部署改善、实际变差次数（|改善| ≥ 2 个积分 SE）、重复间 P5–P95",
             证据="K5：" + K.loc["K5", "关键数字"] + "。K6：" + K.loc["K6", "关键数字"] + "。K9：" + K.loc["K9", "关键数字"],
             证据性质="独立 T2，指定条件"),
    ]
    return pd.DataFrame(rows)


def d3_definitions():
    import pgss_xgb_engine as EN, pgss_r46_t2 as T2
    R = []

    def add(类别, 术语, 定义, 代码=""):
        R.append(dict(类别=类别, 术语=术语, 定义=定义, 代码位置=代码))
    add("任务", "任务与轨", "任务 = 数据库 × 结局 × 二分表型；A 轨 = 表型阳性（Z = 1），B 轨 = 其余。ICU：住院死亡、ICU>72h × 女性、年龄≥65、贫血、eGFR 降低、低白蛋白、肥胖；"
        "NHANES：10 年全因死亡、10 年心脑血管死亡 × 同六表型（O_CVD10×低白蛋白不可执行）", "B1_数据与任务概况")
    add("尺度", "A、B、bal、pop", "R_A、R_B = 轨内平均 Brier；bal = (R_A + R_B)/2（两轨等权，ICU 预定主要尺度）；pop = 按记录等权（人口加权，等于样本中的轨比例）")
    add("尺度", "相对改善", "(R_参照 − R_候选)/R_参照；正值 = 候选更好；不是患者事件风险的降低")
    add("判读", "m 与三态", "m = 1%（技术性实质收益边界，不是 MCID）。支持实质收益 = 下界 > m；排除实质收益 = 上界 < m；其余 = 证据不足。另报相对 0 的方向")
    add("推断", "真实数据区间", "固定 OOF 预测的联合簇重抽（B = 2000，3 个 seed 的逐单位损失先平均）；逐任务区间只作描述；全研究层面的逐轨主张用 max-T 同时带。"
        "不含算法级重新开发的变异", "pgss_r42_inference.py；pgss_r45_nwicu_inference.py；pgss_r44_nhanes_inference.py")
    add("学习程序", "冻结引擎", f"{EN.ENGINE_VERSION}：XGBoost {json.dumps(EN.PARAM_BASE, ensure_ascii=False)}；最多 {EN.MAX_ROUNDS} 轮、耐心 {EN.PATIENCE}；"
        f"早停留出 = 按组 {EN.ES_FOLDS} 折中的 1 折；校准 = 训练内 {EN.CAL_FOLDS} 折交叉拟合 Platt；内层 {EN.INNER_FOLDS} 折", "pgss_xgb_engine.py")
    add("学习程序", "网格", f"专属臂 {EN.GRID_TRACK}；合并臂 {EN.GRID_POOLED}；k ∈ {EN.K_GRID}（平均 |TreeSHAP| 排序，排序模型 {EN.RANK_THETA}）",
        "pgss_xgb_engine.py")
    add("臂", "合并族", f"{EN.POOLED}：S0 = 23 项池、不加 Z；SZ = 加 Z（女性任务 SZ ≡ S0）；SZ-GFS = 全局折内 top-k + Z；SZ-union = 两轨折内所选集合的并集 + Z。"
        f"训练权重 ∈ {EN.WEIGHTS}（bal = 两轨均衡权重）", "pgss_xgb_engine.py")
    add("臂", "专属族", f"{EN.TRACKED}：每轨单独训练；Hfull = 23 项池；HGFS = 全局所选集合；HTFS = 各轨自己的折内选择（k 网格含 23）", "pgss_xgb_engine.py")
    add("臂", "P*", "强合并参照：在 {S0, SZ, SZ-GFS, SZ-union} × {不校准, 校准} × {不加权, 均衡加权} 中取内层交叉验证两轨等权 Brier 最小者（完全相等按所列顺序）",
        "pgss_xgb_engine.develop")
    add("对照", "E1", "HTFS vs P*（预定主要对照）")
    add("对照", "C1、D1（NHANES 追加）", "把合并程序的开发目标（早停评分权重、内层评分、P* 选择、校准开关）改为人口加权（C1：HTFS_pop vs P*_pop）或 B 轨（D1：HTFS_B vs P*_B）；"
        "专属臂参数与面板冻结，只按目标重选校准开关", "pgss_r45_target.py")
    add("模拟", "T2 生成", f"开发集 n = {T2.N_DEV}、π_A = {T2.PI}、每条件 {T2.REPS} 次；条件 {T2.CONDS}；根种子 {T2.ROOT}；结局线性预测子见下行", "pgss_r46_t2.py")
    add("模拟", "T2 结局函数（源码）", inspect.getsource(T2.lin_of).strip(), "pgss_r46_t2.lin_of")
    add("模拟", "T2 评价", f"条件 Brier ℓ = p(1−p) + (p − p̂)² 在独立评价协变量上平均；评价规模逐重复自适应（累计 100k → 500k → 1M，段大小 {T2.STAGES}）；"
        f"边界：候选对 m = {T2.M} 与 0，MIX/POOLMIX 为 0；距边界 < 2 个积分 SE ⇒ 追加；同一重复全部量在同一最终评价集上计算", "pgss_r46_t2.evaluate")
    add("模拟", "D 臂与 F 臂", "P*_D、HTFS_D、S0*_D：用全部开发数据 D 开发；P*_F、HTFS_F、S0*_F：认证划出 1/3 后只用拟合集 F（2/3）开发")
    add("选择器", "ALWAYS_POOLED / ALWAYS_SPLIT", "永远部署 P*_D / 永远部署 HTFS_D（两轨全部专业化）", "pgss_r43_t1.selectors")
    add("选择器", "CV_ANY / CV_POINT_M / CV_LB_M |参照", "参照 ∈ {P*, S0*}；内层 OOF 上：点估计择优 / 相对收益 ≥ m / 近似单侧下界 > m（与认证同一统计量，无保证）；部署 D 臂",
        "pgss_r43_t1.selectors")
    add("选择器", "FCT|参照", f"fit–certify–test：按组划出认证集 f = 1/3；在 F 上开发；认证集上逐轨检验 H0: Δ ≤ m·R(参照)（d_i = (1−m)ℓ_参照 − ℓ_HTFS），"
        f"每轨 α = {T2.ALPHA}；拒绝才对该轨部署 HTFS_F，否则部署参照_F；认证后不重拟合", "pgss_xgb_engine.cert_test；pgss_r43_t1.selectors")
    add("选择器", "USTUN_ADAPT", "Ustun 2019 思路的适配，不称复现：只保留 rationality（认证集上 m = 0 的单侧检验，参照 P*_F，Brier 损失）", "pgss_r43_t1.selectors")
    add("选择器", "MIX", f"逐轨预测混合 (1−t)·P*_D + t·HTFS_D，t ∈ {T2.T_GRID}，取内层 OOF 本轨 Brier 最小（平局取较小 t）", "pgss_r43_t1.selectors")
    add("选择器", "POOLMIX", "逐轨预测混合 (1−t)·P*_D + t·ALT_D；ALT = 另一合并族（P* 为 S0 族时取 SZ*，否则取 S0*）；t 规则同 MIX", "pgss_r46_t2.run_rep")
    add("T2 结果量", "候选对真值相对收益", "D|P* = HTFS_D vs P*_D；D|S0* = HTFS_D vs S0*_D；F|P* = HTFS_F vs P*_F；F|S0* = HTFS_F vs S0*_F")
    add("T2 结果量", "部署改善", "所选部署模型相对 P*_D 的真值相对改善（逐轨；两轨等权 = 两轨风险平均后计算）")
    add("T2 结果量", "实际变差", "部署改善 < 0 且 |改善| ≥ 2 个积分 SE；|改善| < 2 SE 记「边界不确定」", "pgss_r46_t2.run_rep")
    add("T2 结果量", "不达标启用 / 漏启用 / 检出", "启用但所依候选对真值 < m（且不近 m）/ 未启用但真值 ≥ m（且不近 m）/ 条件于确定达标的启用；"
        "重复级「至少一轨不达标启用」按重复计数", "pgss_r46_t2.run_rep")
    add("T2 结果量", "比例的区间", "分子/分母、Clopper–Pearson 95% 区间、Monte Carlo 标准误；0/200 只报 0/200 与上界，不写成错误率已验证低于某值")
    add("T2 结果量", "均值的区间", "200 次重复的均值 ± 1.96·重复间标准误；本包「均值±1.96SE方向」据此分为区间整体 > 0 / < 0 / 跨 0 / 全部重复为 0")
    return pd.DataFrame(R)


def d4_literature():
    t = rd("文献表", "先例原文定位")
    TH = ["信息提供", "架构比较", "目标一致性", "选择成本"]
    rows = []
    for k, h in t.groupby("编号", sort=False):
        r = dict(编号=k, 文献=h.文献.iloc[0], 标识=h.标识.iloc[0])
        for th in TH:
            x = h[h.主题 == th]
            r[th] = "；".join(f"原文（{a.原文位置}）：“{a.原文摘录}” → {a.与本研究的关系}" for a in x.itertuples()) if len(x) else "本表未摘录（未核对）"
        r["核对级别"] = "；".join(sorted(set(h.核对级别)))
        r["摘录条数"] = len(h)
        rows.append(r)
    return pd.DataFrame(rows)


# ── 图源（F1–F6）───────────────────────────────────────────────
PH_EN = {"女性": "Female", "年龄≥65": "Age ≥65", "贫血": "Anemia", "eGFR降低": "Reduced eGFR", "低白蛋白": "Low albumin", "肥胖": "Obesity"}
OC_EN = {"OUT_HOSP_MORT": "In-hospital death", "OUT_LOS72": "ICU stay >72 h", "O_MORT10": "10-y all-cause death", "O_CVD10": "10-y CVD death"}
DB_EN = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU", "nhanes": "NHANES"}


def f_sources(S):
    F = {}
    B1, B2 = S["B1_数据与任务概况"], S["B2_三层总结果表"]
    n = B1.groupby("库").agg(任务=("纳入", "size"), 纳入任务=("纳入", "sum")).reset_index()
    F["F1_研究设计"] = pd.concat([
        n.assign(块=n.库.map(lambda k: "L2 cross-domain replication" if k == "nhanes" else "L1 ICU benchmark"), 标签=n.库.map(DB_EN),
                 数值=n.纳入任务, 单位="included tasks"),
        pd.DataFrame([dict(块="L3 post hoc objective-alignment control", 标签="NHANES low albumin", 数值=1, 单位="task"),
                      dict(块="L1 ICU benchmark", 标签="outcomes per database", 数值=int(B1[B1.库 == "mimic"].结局.nunique()), 单位="outcomes"),
                      dict(块="L1 ICU benchmark", 标签="phenotypes per database", 数值=int(B1[B1.库 == "mimic"].表型.nunique()), 单位="phenotypes"),
                      dict(块="T2 independent simulation", 标签="conditions", 数值=len(S["C1_T2条件与设计"]), 单位="conditions"),
                      dict(块="T2 independent simulation", 标签="selection procedures", 数值=int(S["C3_T2两轨等权部署"].选择器.nunique()), 单位="procedures"),
                      dict(块="T2 independent simulation", 标签="replicates per condition", 数值=int(S["C1_T2条件与设计"].完成重复.iloc[0]),
                           单位="replicates")])], ignore_index=True)[["块", "标签", "数值", "单位"]]
    f2 = B2[B2.层.str.startswith("L1") & (B2.尺度 == "bal")].copy()
    f2["Database"] = f2.库.map(DB_EN); f2["Outcome"] = f2.结局.map(OC_EN); f2["Phenotype"] = f2.表型.map(PH_EN)
    f2["ph_order"] = f2.表型.map({p: i for i, p in enumerate(PH_ORDER)})
    F["F2_ICU复现"] = f2.sort_values(["库", "结局", "ph_order"])[["Database", "Outcome", "Phenotype", "库", "结局", "表型", "观察相对改善", "相对_P2_5", "相对_P97_5",
                                                                "三态_m1", "相对0方向"]].reset_index(drop=True)
    b5 = S["B5_NHANES低白蛋白归因"]
    k = b5[((b5.对照 == "E1") & b5.尺度.isin(["B", "pop"])) | ((b5.对照 == "D1") & (b5.尺度 == "B")) | ((b5.对照 == "C1") & (b5.尺度 == "pop"))].copy()
    k["Scale"] = k.尺度.map({"B": "Track B (non-low-albumin)", "pop": "Population-weighted"})
    k["Comparison"] = k.对照.map({"E1": "Original: HTFS vs P* (P* developed for balanced objective)",
                                  "D1": "Objective-aligned: HTFS_B vs P*_B", "C1": "Objective-aligned: HTFS_pop vs P*_pop"})
    F["F3a_NHANES归因_相对"] = k[["Scale", "Comparison", "对照", "尺度", "观察相对改善", "相对_P2_5", "相对_P97_5", "三态_m1"]].reset_index(drop=True)
    g = S["B6_NHANES差距收回"]
    F["F3b_NHANES归因_绝对"] = g.melt(id_vars=["目标", "尺度"], value_vars=["R_P*（两轨等权开发）", "R_P*_目标", "R_HTFS_目标", "R_HTFS（原）"],
                                    var_name="模型", value_name="Brier")
    lc = rd("R42推断", "8_学习曲线"); full = rd("R42推断", "2_全部对照")
    full = full[(full.对照 == "E1") & (full.尺度 == "bal")][["库", "结局", "表型", "观察相对改善"]].rename(columns={"观察相对改善": "HTFS_vs_P*_bal_相对"})
    full["比例"] = 1.0; full["完整度"] = 1.0
    part = lc[["库", "结局", "表型", "比例", "完整度", "HTFS_vs_P*_bal_相对"]]
    f4 = pd.concat([part, full], ignore_index=True)
    f4["可画"] = f4.完整度 >= 0.999
    f4["Database"] = f4.库.map(DB_EN); f4["Outcome"] = f4.结局.map(OC_EN); f4["Phenotype"] = f4.表型.map(PH_EN)
    F["F4_学习曲线"] = f4.sort_values(["库", "结局", "表型", "比例"]).reset_index(drop=True)
    c2 = S["C2_T2候选对真值收益"]
    F["F5_T2机制"] = c2[c2.候选对 == "D|P*"][["条件", "轨", "重复", "真值相对收益均值", "SE", "下", "上", "P5", "P95", "相对m方向"]].reset_index(drop=True)
    c3 = S["C3_T2两轨等权部署"]
    F["F6a_T2部署两轨等权"] = c3[c3.选择器.isin(SELS)][["条件", "选择器", "两轨等权部署改善均值", "SE", "下", "上", "均值±1.96SE方向"]].reset_index(drop=True)
    c4 = S["C4_T2逐轨部署与变差"]
    F["F6b_T2逐轨实际变差"] = c4[c4.选择器.isin(SELS)][["条件", "选择器", "轨", "部署实际变差_分子", "部署实际变差_分母", "部署实际变差_CP下",
                                                     "部署实际变差_CP上"]].reset_index(drop=True)
    c6 = S["C6b_T2重复级启用错误"]
    F["F6c_T2启用错误"] = c6[c6.选择器.isin(["FCT|P*", "CV_POINT_M|P*"])][["条件", "选择器", "≥1轨不达标启用_分子", "≥1轨不达标启用_分母",
                                                                        "≥1轨不达标启用_CP下", "≥1轨不达标启用_CP上"]].reset_index(drop=True)
    return F


FIGS = [
    ("Fig1", "R47_Fig1_study_design", "研究设计：三层真实数据证据与独立模拟（示意，数字来自 F1 源表）", "F1_研究设计"),
    ("Fig2", "R47_Fig2_ICU_benchmark", "三个 ICU 库 36 个任务：HTFS 相对 P* 的两轨等权相对 Brier 改善与 95% 区间（固定 OOF 簇重抽）；竖线 0 与 m = 1%", "F2_ICU复现"),
    ("Fig3", "R47_Fig3_NHANES_attribution", "NHANES 低白蛋白（10 年全因死亡）：a 原比较与目标一致对照的相对改善（B 轨、人口加权）；b 各模型的绝对 Brier",
     "F3a_NHANES归因_相对；F3b_NHANES归因_绝对"),
    ("Fig4", "R47_Fig4_learning_curves", "MIMIC-IV 与 eICU-CRD 学习曲线：HTFS 相对 P* 的两轨等权相对改善随目标训练比例（按簇）变化；只画全部外层折可评价的点", "F4_学习曲线"),
    ("Fig5", "R47_Fig5_T2_mechanisms", "T2：8 个条件下 HTFS_D 相对 P*_D 的逐轨真值相对收益（均值 ± 1.96·SE 与 P5–P95）；虚线 m = 1%", "F5_T2机制"),
    ("Fig6", "R47_Fig6_T2_selection", "T2：a 选择器两轨等权部署改善（相对 P*_D）；b A 轨与 B 轨实际变差的重复比例；c 重复级至少一轨不达标启用（CP 区间）",
     "F6a_T2部署两轨等权；F6b_T2逐轨实际变差；F6c_T2启用错误"),
]


def d5_figs(F):
    rows = []
    for k, fn, desc, src in FIGS:
        n = sum(len(F[s.strip()]) for s in src.split("；"))
        rows.append(dict(图=k, 文件=f"figures/{fn}.png；figures/{fn}.pdf", 内容=desc, 源表=src, 源表行数=n))
    return pd.DataFrame(rows)


def d6_tables():
    T = [("主表 1（建议）", "B1_数据与任务概况", "各库任务的人数、事件与轨构成"),
         ("主表 2（建议）", "D2_评价问题与证据", "GPT 第 51 轮 §四 的评价顺序与本研究中的估计量、证据"),
         ("补充表", "B2_三层总结果表", "三层真实数据结果全表（E1 四个尺度；C1、D1）"),
         ("补充表", "B3_分层计数", "三层 × 库 × 尺度的三态与方向计数"), ("补充表", "B4_maxT汇总", "逐轨 max-T 同时判断"),
         ("补充表", "B5_NHANES低白蛋白归因", "NHANES 低白蛋白：E1、C1–C3、D1–D3 全部尺度"), ("补充表", "B6_NHANES差距收回", "绝对 Brier 与差距收回"),
         ("补充表", "C1_T2条件与设计", "T2 条件、生成与评价规模"), ("补充表", "C2_T2候选对真值收益", "四个候选对 × 两轨"),
         ("补充表", "C3_T2两轨等权部署", "全部选择器"), ("补充表", "C4_T2逐轨部署与变差", "全部选择器 × 两轨"),
         ("补充表", "C5_MIX等三面并列", "平均、逐轨变差与变异"), ("补充表", "C5b_MIX与POOLMIX配对差", "同重复配对差"), ("补充表", "C6a_T2逐轨启用与检出", "认证与 CV 规则的逐轨启用、不达标启用与检出"),
         ("补充表", "C6b_T2重复级启用错误", "重复级至少一轨"), ("补充表", "C6c_T2认证样本成本", "P*_F 相对 P*_D"),
         ("补充表", "C7_T1与T2数值并列", "T1 探索与 T2 采用"), ("补充表", "C8_两两排序反转_描述", "描述性"), ("补充表", "C9_T2极端值_初始10万", "生成分布诊断"),
         ("补充表", "C10_plasmode探索", "R45 plasmode（探索性）"), ("方法补充", "D3_方法定义", "术语、估计量、程序与冻结常量的唯一来源"),
         ("相关工作", "D4_文献比较", "直接相关原始研究（本地原文逐字核对的摘录）")]
    return pd.DataFrame(T, columns=["建议位置", "表", "内容"])


def d7_entries():
    E = [("数据 v5 构建（MIMIC/eICU）", "pgss_r383_build_v5.py", "受限数据（PostgreSQL）", "冻结 data v5"),
         ("ICU 主实验（main + cert + 学习曲线）", "pgss_xgb_run.py + pgss_xgb_engine.py", "data v5", "R42 逐单元产物（患者级预测只留本地）"),
         ("ICU 推断", "pgss_r42_inference.py", "R42 逐单元产物", "R42_INFERENCE.xlsx"),
         ("NWICU 队列、运行与推断", "pgss_r45_nwicu_build.py、pgss_r45_nwicu_run.py、pgss_r45_nwicu_inference.py；结局修正 pgss_r46_outcome_verify.py、pgss_r46_rerun.py、pgss_r46_inference.py",
          "受限数据", "nwicu_v2 推断"),
         ("NHANES 队列、运行与推断", "pgss_r44_nhanes_cohort.py、pgss_r44_nhanes_run.py、pgss_r44_nhanes_inference.py（死因修正同上）", "NHANES 公共文件（本地）", "nhanes_v2 推断"),
         ("NHANES 目标一致性追加对照", "pgss_r45_target.py、pgss_r45_target_inference.py", "NHANES", "target_v2 推断"),
         ("T1（探索）", "pgss_r43_t1.py、pgss_r43_t1_summary.py", "无（合成）", "R43 T1 汇总"),
         ("plasmode（探索）", "pgss_r45_plasmode.py、pgss_r45_plasmode_summary.py", "受限数据（真实协变量）", "R45 plasmode 汇总"),
         ("T2", "pgss_r46_t2.py、pgss_r46_t2_summary.py", "无（合成）", "T2 逐重复记录与汇总"),
         ("合成示例（本轮新增入口）", "pgss_r47_example.py", "无（合成）", "examples/ 下的示例输出（不进入论文）"),
         ("冻结数字包、图与报告", "pgss_r47_numbers.py、pgss_r47_figures.py、pgss_r47_report.py", "已交付结果表", "R47 三件"),
         ("打包与自检", "pgss_r47_package.py、pgss_r47_package_tests.py", "—", "R47_CODE_AND_FIGURES.zip")]
    return pd.DataFrame(E, columns=["步骤", "脚本", "输入", "输出"])


# ── 汇总 ─────────────────────────────────────────────────────────
def corrections(S, ext_line):
    C7, C3, C8, C5, C6b = S["C7_T1与T2数值并列"], S["C3_T2两轨等权部署"], S["C8_两两排序反转_描述"], S["C5_MIX等三面并列"], S["C6b_T2重复级启用错误"]
    cg = C7[C7.T1与T2 == "T2 区间支持的方向与 T1 均值异号"]
    f = C3[C3.选择器 == "FCT|P*"]
    nn = int((f["均值±1.96SE方向"] == "区间整体 < 0").sum()); cr = f[f["均值±1.96SE方向"] == "区间跨 0"]
    m1 = C5[(C5.选择器 == "MIX") & (C5.条件 == "N1")].iloc[0]
    nul = C6b[(C6b.选择器 == "FCT|P*") & C6b.条件.isin(["N1", "N2", "N1-T", "N1-H"])]
    rows = [
        dict(编号="R1", GPT要求="不以「Q1–Q5 推翻 0」替代 T1/T2 的实际数值对比", R46原处="报告 §0 第 3 条、§4.8、§9 意见 4；表 D11",
             R47做法="C7 逐项数值并列 T1 与 T2（每行给 T2 均值 ± 1.96·SE 的方向与 T1 均值符号）；Q1–Q5 计数不再作为概括",
             依据表="C7_T1与T2数值并列", 结果=f"{len(C7)} 行中，T2 区间支持的方向与 T1 均值异号 {len(cg)} 行"),
        dict(编号="R2", GPT要求="区分负点估计与区间支持的负向结果", R46原处="§9 意见 5「两轨等权部署改善在 8/8 个条件为负」",
             R47做法="C2–C5 加「均值±1.96SE方向」列；矩阵 K8 分开写", 依据表="C3_T2两轨等权部署",
             结果=f"FCT|P* 两轨等权：区间整体 < 0 {nn}/8；区间跨 0：" + "、".join(f"{r.条件}（{p2(r.下)}～{p2(r.上)}）" for r in cr.itertuples())),
        dict(编号="R3", GPT要求="极端值与尾部比例标明来自初始 10 万评价样本", R46原处="表 D1 的极端值列（列名未标评价范围）",
             R47做法="C9 列名加前缀「初始10万评价样本_」并加「评价范围」列；不重新生成评价样本", 依据表="C9_T2极端值_初始10万",
             结果=f"生成代码中计算极端值的语句只用第 0 段 Ev0：{ext_line[:90]}…" if ext_line else "未在代码中找到该语句"),
        dict(编号="R4", GPT要求="两两排序反转保持描述性", R46原处="§4.8、§9 意见 4（25 个选择器对）",
             R47做法="C8 首列注明比较总数与未作多重调整", 依据表="C8_两两排序反转_描述", 结果=f"{len(C8)}/{S['_n_cmp']}"),
        dict(编号="R5", GPT要求="FCT 的模拟错误率证据与普适保证分开", R46原处="§4.4、§9 意见 5",
             R47做法="C6a、C6b 首列注明「指定生成条件下 200 次重复的 Monte Carlo 估计；不是错误率保证」", 依据表="C6b_T2重复级启用错误",
             结果=f"无结构条件的 FCT|P* 重复级至少一轨不达标启用 {int(nul['≥1轨不达标启用_分子'].min())}–{int(nul['≥1轨不达标启用_分子'].max())}/200，"
                  f"CP 上界最大 {p2(nul['≥1轨不达标启用_CP上'].max())}"),
        dict(编号="R6", GPT要求="MIX 的平均改善、逐轨风险恶化和变异同时呈现", R46原处="§4.5、§9 意见 6",
             R47做法="C5 每行并列两轨等权均值与方向、P5–P95、逐轨均值与方向、逐轨实际变差次数", 依据表="C5_MIX等三面并列",
             结果=f"N1 MIX：两轨等权 {p2(m1.两轨等权均值)}，A 轨 {p2(m1.A轨均值)}，A 轨实际变差 {m1.A轨实际变差}/{m1.A轨重复}"),
    ]
    return pd.DataFrame(rows)


def build():
    TRACE.clear()
    S = {}
    S["A2_输入与交付副本比对"] = a2_inputs()
    S["B1_数据与任务概况"] = b1_tasks()
    S["B2_三层总结果表"] = ev("F5_三层总结果表")
    S["B3_分层计数"] = ev("F1_三层计数")
    S["B4_maxT汇总"] = ev("F2_maxT汇总")
    S["B5_NHANES低白蛋白归因"] = b5_nhanes_case()
    S["B6_NHANES差距收回"] = b6_gap()
    S["C1_T2条件与设计"] = c1_design()
    S["C2_T2候选对真值收益"] = c2_pairs()
    S["C3_T2两轨等权部署"] = c3_bal()
    S["C4_T2逐轨部署与变差"] = c4_track()
    S["C5_MIX等三面并列"] = c5_mix(S["C3_T2两轨等权部署"], S["C4_T2逐轨部署与变差"])
    g = ev("D6_T2_MIX与POOLMIX")
    S["C5b_MIX与POOLMIX配对差"] = g[g.选择器.astype(str).str.startswith("MIX − POOLMIX")][["条件", "选择器", "重复", "风险变化均值", "SE", "P5", "中位", "P95",
                                                                                     "MIX更好重复"]].reset_index(drop=True)
    S["A3_T2逐重复记录导出"] = export_t2_records()
    S["C6a_T2逐轨启用与检出"], S["C6b_T2重复级启用错误"], S["C6c_T2认证样本成本"] = c6_fct()
    S["C7_T1与T2数值并列"] = c7_t1t2()
    S["C8_两两排序反转_描述"], S["_n_cmp"] = c8_rank()
    S["C9_T2极端值_初始10万"], ext_line = c9_extreme()
    S["C10_plasmode探索"] = c10_plasmode()
    S["D1_主张证据限制矩阵"] = d1_matrix(S)
    S["D2_评价问题与证据"] = d2_questions(S)
    S["D3_方法定义"] = d3_definitions()
    S["D4_文献比较"] = d4_literature()
    F = f_sources(S)
    S["D5_图索引"] = d5_figs(F)
    S["D6_表格索引"] = d6_tables()
    S["D7_复现入口"] = d7_entries()
    S["A1_报告层更正"] = corrections(S, ext_line)
    S.update(F)
    S["_ext_line"] = ext_line
    return S


XL = OUT / "R47_冻结数字包.xlsx"
RECS = OUT / "t2_records"


def export_t2_records():
    """T2 逐重复记录（合成数据）：parquet 原样取自已交付的 R46 zip；另写 CSV（%.17g，round_trip 读回逐位相同），供没有 parquet 引擎的复核环境。
    核对：CSV 读回与 parquet 逐位相同；变异对照：读回值乘 1 + 1e-15 后必须被判为不同。"""
    import io
    RECS.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(Z46); rows = []
    (RECS / "rep.jsonl").write_bytes(z.read("r46/t2/runs_records/rep.jsonl"))
    for t in ("arms", "pairs", "sel"):
        b = z.read(f"r46/t2/runs_records/{t}.parquet"); (RECS / f"{t}.parquet").write_bytes(b)
        d = pd.read_parquet(io.BytesIO(b))
        d.to_csv(RECS / f"{t}.csv.gz", index=False, float_format="%.17g", compression="gzip")
        r = pd.read_csv(RECS / f"{t}.csv.gz", float_precision="round_trip")
        fc = [c for c in d.columns if d[c].dtype.kind == "f"]
        same = all(np.array_equal(d[c].to_numpy(), r[c].to_numpy(float), equal_nan=True) for c in fc) and list(d.columns) == list(r.columns) \
            and len(d) == len(r)
        mut = r[fc[0]].to_numpy(float) * (1 + 1e-15)
        caught = not np.array_equal(d[fc[0]].to_numpy(), mut, equal_nan=True)
        rows.append(dict(表=t, 行数=len(d), 浮点列=len(fc), CSV读回与parquet逐位相同=bool(same), 变异对照检出=bool(caught),
                         parquet_SHA16=sha16(b), csv_gz_SHA16=sha16((RECS / f"{t}.csv.gz").read_bytes())))
    n = sum(1 for l in (RECS / "rep.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    rows.append(dict(表="rep.jsonl", 行数=n, 浮点列=np.nan, CSV读回与parquet逐位相同=np.nan, 变异对照检出=np.nan,
                     parquet_SHA16=sha16((RECS / "rep.jsonl").read_bytes()), csv_gz_SHA16=""))
    return pd.DataFrame(rows)
ORDER_NOTE = {"A": "更正与输入核对", "B": "真实数据三层", "C": "T2 与模拟", "D": "写作资产", "F": "图源数据（图中每个点）"}


def sheets(S):
    return {k: v for k, v in S.items() if isinstance(v, pd.DataFrame)}


def write_xlsx(S, cited, path=XL, note=None):
    """目录（表、类别、报告引用）+ 0_说明 + 全部表；cited = md 中引用数字的表。"""
    sh = sheets(S)
    nat = lambda k: ("ABCDF".index(k[0]), int(re.match(r"[A-Z](\d+)", k).group(1)), k)
    order = sorted(sh, key=nat)
    toc = pd.DataFrame([dict(表=k, 类别=ORDER_NOTE[k[0]], 行数=len(sh[k]), 报告引用="是" if k in cited else "否") for k in order])
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        toc.to_excel(xw, sheet_name="目录", index=False)
        pd.DataFrame({"说明": note or []}).to_excel(xw, sheet_name="0_说明", index=False)
        for k in order:
            sh[k].to_excel(xw, sheet_name=k[:31], index=False)
    return toc


if __name__ == "__main__":
    S = build()
    for k, v in S.items():
        if isinstance(v, pd.DataFrame):
            print(k, v.shape)
    print(S["A2_输入与交付副本比对"].to_string())
    print(S["A1_报告层更正"].to_string())
