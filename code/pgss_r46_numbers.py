# -*- coding: utf-8 -*-
r"""R46 写作数字包（GPT 第 50 轮 §5.2、§8）：三层总结果表、分层计数、max-T 汇总、主图源数据、主张—证据—限制矩阵。只读结果表，不拟合。

三层（不合并成没有明确估计对象的总效应）：
  L1 ICU 预定两轨等权比较：MIMIC-IV、eICU-CRD（R42）、NWICU（R46 v2，住院死亡结局已修正）
  L2 NHANES 原始跨领域复现（R44 v2：O_CVD10 死因未知一例已修正）
  L3 NHANES 结果知情后登记的目标一致性追加对照（R45 v2：C1 pop 尺度 11 任务；D1 B 尺度 1 任务）
证据层级标签：预定主要比较 / 跨领域复现 / 结果知情后追加对照 / 探索性模拟（T1、plasmode）/ 独立 T2。
矩阵中的数字全部由本脚本从结果表取出；「主张」一栏是可检验的候选表述与适用范围，不是论文正文。
输出：28_R46_最终核验与T2/numbers/R46_NUMBERS.xlsx、.json
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
R46 = ROOT / "28_R46_最终核验与T2"
OUT = R46 / "numbers"
R42X = ROOT / "23_R42_正式主实验" / "inference" / "R42_INFERENCE.xlsx"
NWX = R46 / "nwicu_v2" / "inference" / "R45_NWICU_INFERENCE.xlsx"
NHX = R46 / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx"
TGX = R46 / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx"
ROX = R46 / "readonly" / "R46_READONLY.xlsx"
R43RO = ROOT / "24_R43_结果补齐与T1" / "readonly" / "R43_READONLY.xlsx"
T2X = R46 / "t2" / "R46_T2_SUMMARY.xlsx"
PLX = ROOT / "27_R45_外部复现与机制核验" / "plasmode" / "R45_PLASMODE_SUMMARY.xlsx"
COLS = ["层", "库", "结局", "表型", "对照", "参照", "候选", "尺度", "观察相对改善", "相对_P2_5", "相对_P97_5", "三态_m1", "相对0方向"]


def std(df):
    """统一列名（R42 用 相对_P2.5，其余用 相对_P2_5）。"""
    df = df.rename(columns={"相对_P2.5": "相对_P2_5", "相对_P97.5": "相对_P97_5", "三态_m0.010": "三态_m1"})
    return df


def layers():
    r42 = std(pd.read_excel(R42X, sheet_name="2_全部对照")); r42 = r42[r42.对照 == "E1"]
    nw = std(pd.read_excel(NWX, sheet_name="2_Brier对照")); nw = nw[nw.对照 == "E1"]
    nh = std(pd.read_excel(NHX, sheet_name="2_Brier对照")); nh = nh[nh.对照 == "E1"]
    tg = std(pd.read_excel(TGX, sheet_name="2_Brier对照"))
    tg = tg[((tg.对照 == "C1") & (tg.尺度 == "pop")) | ((tg.对照 == "D1") & (tg.尺度 == "B"))]
    L = pd.concat([r42.assign(层="L1 ICU 预定两轨等权比较"), nw.assign(层="L1 ICU 预定两轨等权比较", 库="nwicu"),
                   nh.assign(层="L2 NHANES 原始跨领域复现", 库="nhanes"),
                   tg.assign(层="L3 NHANES 结果知情后目标一致性追加对照", 库="nhanes")], ignore_index=True)
    return L[COLS]


def counts(L):
    rows = []
    for (ly, db, c, sc), h in L.groupby(["层", "库", "对照", "尺度"], sort=False):
        t = h.三态_m1.value_counts(); d = h.相对0方向.value_counts()
        rows.append(dict(层=ly, 库=db, 对照=c, 尺度=sc, 任务=len(h), 排除=int(t.get("排除实质收益", 0)), 证据不足=int(t.get("证据不足", 0)),
                         支持=int(t.get("支持实质收益", 0)), 全负=int(d.get("全负", 0)), 跨零=int(d.get("跨零", 0)), 全正=int(d.get("全正", 0)),
                         相对改善中位=float(h.观察相对改善.median()), 点估计为负=int((h.观察相对改善 < 0).sum()),
                         相对上界最大=float(h.相对_P97_5.max())))
    return pd.DataFrame(rows)


def maxt():
    r42 = pd.read_excel(R42X, sheet_name="3_E1_maxT")
    parts = [r42.assign(来源="R42")]
    for f, sh, db in ((NWX, "5_E1_maxT", "nwicu"), (NHX, "5_E1_maxT", "nhanes")):
        parts.append(pd.read_excel(f, sheet_name=sh).assign(库=db, 来源="R46 v2"))
    M = pd.concat(parts, ignore_index=True)
    rows = []
    for db, h in M.groupby("库", sort=False):
        v = h.同时判定_相对m.value_counts()
        rows.append(dict(库=db, 项数=len(h), 排除=int(sum(c for k, c in v.items() if str(k).startswith("排除"))),
                         证据不足=int(v.get("证据不足", 0)), 支持=int(sum(c for k, c in v.items() if str(k).startswith("支持"))),
                         max_T临界值=float(h["max-T临界值"].iloc[0]),
                         证据不足项="；".join(f"{r.结局}×{r.表型}×{r.轨}" for r in h.itertuples() if r.同时判定_相对m == "证据不足")))
    tg = pd.read_excel(TGX, sheet_name="5_C1_maxT")
    v = tg.同时判定_相对m.value_counts()
    rows.append(dict(库="nhanes（C1 pop，追加对照）", 项数=len(tg), 排除=int(sum(c for k, c in v.items() if str(k).startswith("排除"))),
                     证据不足=int(v.get("证据不足", 0)), 支持=int(sum(c for k, c in v.items() if str(k).startswith("支持"))),
                     max_T临界值=float(tg["max-T临界值"].iloc[0]),
                     证据不足项="；".join(f"{r.结局}×{r.表型}" for r in tg.itertuples() if r.同时判定_相对m == "证据不足")))
    return pd.DataFrame(rows), M


def fig_sources(L):
    F = {}
    F["F2_ICU复现"] = L[L.层.str.startswith("L1")].copy()
    tg = std(pd.read_excel(TGX, sheet_name="2_Brier对照"))
    k = (tg.结局 == "O_MORT10") & (tg.表型 == "低白蛋白")
    F["F3_NHANES目标一致性_对照"] = tg[k][["对照", "参照", "候选", "尺度", "观察相对改善", "相对_P2_5", "相对_P97_5", "三态_m1"]]
    gp = pd.read_excel(TGX, sheet_name="6_差距收回")
    F["F3_NHANES目标一致性_风险"] = gp[(gp.结局 == "O_MORT10") & (gp.表型 == "低白蛋白")]
    F["F4_学习曲线"] = pd.read_excel(R43RO, sheet_name="11_学习曲线绝对风险")
    if T2X.exists():
        tc = pd.read_excel(T2X, sheet_name="C_候选对真值收益")
        F["F5_已知机制确认_T2"] = tc
        F["F5_已知机制确认_T1并列"] = pd.read_excel(T2X, sheet_name="I_与T1并列")
        td = pd.read_excel(T2X, sheet_name="D_选择器逐轨"); te = pd.read_excel(T2X, sheet_name="E_选择器两轨等权")
        F["F6_选择错误与部署风险_逐轨"] = td
        F["F6_选择错误与部署风险_两轨等权"] = te
    return F


def t2_side(r, m=0.01):
    """候选对真值收益的措辞由数据决定：均值 ± 1.96·SE 相对 m 与 0 的位置。"""
    lo, hi = r.真值相对收益均值 - 1.96 * r.SE, r.真值相对收益均值 + 1.96 * r.SE
    if lo >= m:
        return "均值区间整体 ≥ m"
    if lo > 0:
        return "均值区间整体 > 0、但跨 m" if hi >= m else "均值区间整体在 0 与 m 之间"
    if hi < 0:
        return "均值区间整体 < 0"
    return "均值区间跨 0"


def matrix(L, C, MX):
    """主张—证据—限制矩阵：数字全部来自 C（分层计数）与 MX（max-T）及 T2 汇总。"""
    g = lambda ly, db, c, sc: C[(C.层.str.startswith(ly)) & (C.库 == db) & (C.对照 == c) & (C.尺度 == sc)].iloc[0]
    rows = []
    for db in ("mimic", "eicu", "nwicu"):
        r = g("L1", db, "E1", "bal"); m = MX[MX.库 == db].iloc[0]
        rows.append(dict(编号=f"M-{db}", 证据层级="预定主要比较" if db != "nwicu" else "独立复现（预定主要比较的同一程序）",
                         主张=f"{db}：两轨等权尺度上，HTFS 相对 P* 没有任务显示被区间支持的 ≥1% 增量", 估计对象="固定 OOF 预测的两轨等权 Brier 相对改善（E1）",
                         关键数字=f"{r.任务} 任务：排除 {r.排除}、证据不足 {r.证据不足}、支持 {r.支持}；点估计为负 {r.点估计为负}；中位 {100 * r.相对改善中位:.2f}%；"
                                  f"上界最大 {100 * r.相对上界最大:.2f}%；逐轨 max-T {m.项数} 项：排除 {m.排除}、证据不足 {m.证据不足}、支持 {m.支持}",
                         来源="N1/N2/N3", 限制="区间是固定 OOF 预测的簇重抽（不是算法级重新开发的置信保证）；逐任务区间与同时判断分开陈述"
                                                + ("；eICU 有效独立簇少" if db == "eicu" else "") + ("；住院死亡结局在看到 R45 结果后修正（UNKNOWN 去向）" if db == "nwicu" else "")))
    r = g("L2", "nhanes", "E1", "bal"); m = MX[MX.库 == "nhanes"].iloc[0]
    rB, rP = g("L2", "nhanes", "E1", "B"), g("L2", "nhanes", "E1", "pop")
    rows.append(dict(编号="M-nhanes-v10", 证据层级="跨领域复现", 主张="NHANES：两轨等权尺度上没有任务被支持；个别任务在 B 轨或人口加权尺度上原比较显示优势",
                     估计对象="同上（E1，两轨等权开发的 P*）",
                     关键数字=f"{r.任务} 任务（bal）：排除 {r.排除}、证据不足 {r.证据不足}、支持 {r.支持}；B 轨支持 {rB.支持}、人口加权支持 {rP.支持}；"
                              f"逐轨 max-T {m.项数} 项：排除 {m.排除}、证据不足 {m.证据不足}、支持 {m.支持}",
                     来源="N1/N2/N3", 限制="一般人群、长期死亡结局；不是未触碰数据；不使用调查权重"))
    c1 = L[(L.对照 == "C1") & (L.结局 == "O_MORT10") & (L.表型 == "低白蛋白")].iloc[0]
    d1 = L[(L.对照 == "D1")].iloc[0]
    rows.append(dict(编号="M-nhanes-target", 证据层级="结果知情后追加对照", 主张="低白蛋白任务：专属模型相对原合并程序的优势，没有在目标一致的强合并参照之上保留下来",
                     估计对象="C1：HTFS_pop vs P*_pop（人口加权=研究样本比例）；D1：HTFS_B vs P*_B（B 轨）",
                     关键数字=f"C1 {100 * c1.观察相对改善:.2f}%（{100 * c1.相对_P2_5:.2f}%～{100 * c1.相对_P97_5:.2f}%，{c1.三态_m1}）；"
                              f"D1 {100 * d1.观察相对改善:.2f}%（{100 * d1.相对_P2_5:.2f}%～{100 * d1.相对_P97_5:.2f}%，{d1.三态_m1}）",
                     来源="N1、F3", 限制="在 R44 结果后登记；只有一个任务有预定解释性判读；不证明差异精确为零；不分解早停、参数、面板与校准各自的贡献"))
    if PLX.exists():
        pb = pd.read_excel(PLX, sheet_name="B_候选对真值收益"); pdd = pd.read_excel(PLX, sheet_name="D_选择器两轨等权")
        for dl in (0.0, 1.0):
            a = pb[(pb.δ == dl) & (pb.候选对 == "D|P*") & (pb.轨 == "A")].iloc[0]; b = pb[(pb.δ == dl) & (pb.候选对 == "D|P*") & (pb.轨 == "B")].iloc[0]
            mx = pdd[(pdd.δ == dl) & (pdd.选择器 == "MIX")].iloc[0]; fc = pdd[(pdd.δ == dl) & (pdd.选择器 == "FCT|P*")].iloc[0]
            rows.append(dict(编号=f"M-plasmode-δ{dl:g}", 证据层级="探索性模拟（plasmode，R45）",
                             主张=f"δ = {dl:g}：真实 MIMIC 协变量结构上的已知生成机制中，专属模型相对强合并参照的真值收益与选择程序的部署结果",
                             估计对象="条件于该次固定评价协变量和给定生成概率的期望 Brier 精确值",
                             关键数字=f"HTFS_D vs P*_D：A {100 * a.真值相对收益均值:.2f}%、B {100 * b.真值相对收益均值:.2f}%（各 {int(a.重复)} 次）；"
                                      f"两轨等权部署改善 MIX {100 * mx.均值:.2f}%、FCT|P* {100 * fc.均值:.2f}%",
                             来源="R45 plasmode 汇总", 限制="只对所设生成机制成立；δ 不对应临床异质性量表；不说明真实死亡结局中的轨道特异结构强弱"))
    if T2X.exists():
        tc = pd.read_excel(T2X, sheet_name="C_候选对真值收益"); te = pd.read_excel(T2X, sheet_name="E_选择器两轨等权")
        tf = pd.read_excel(T2X, sheet_name="F_重复级至少一轨")
        for cond in ("P2", "P2-T", "P2-H"):
            h = tc[(tc.条件 == cond) & (tc.候选对 == "D|P*")]
            if h.empty:
                continue
            ga, gb = h[h.轨 == "A"].iloc[0], h[h.轨 == "B"].iloc[0]
            mix = te[(te.条件 == cond) & (te.选择器 == "MIX")].iloc[0]; fct = te[(te.条件 == cond) & (te.选择器 == "FCT|P*")].iloc[0]
            rows.append(dict(编号=f"M-T2-{cond}", 证据层级="独立 T2" + ("（分布压力条件）" if cond.endswith("-H") else ""),
                             主张=f"{cond}：已知轨道特异结构下，冻结程序的专属模型相对强合并参照的真值收益——A 轨{t2_side(ga)}，B 轨{t2_side(gb)}；"
                                  "MIX 与 FCT|P* 的两轨等权部署改善见关键数字",
                             估计对象="条件 Brier 真值（评价协变量上的 Monte Carlo 积分）",
                             关键数字=f"HTFS_D vs P*_D：A {100 * ga.真值相对收益均值:.2f}%（SE {100 * ga.SE:.2f}%）、B {100 * gb.真值相对收益均值:.2f}%（SE {100 * gb.SE:.2f}%）；"
                                      f"两轨等权部署改善 MIX {100 * mix.均值:.2f}%、FCT|P* {100 * fct.均值:.2f}%（各 {int(mix.重复)} 次）",
                             来源="N8、N9", 限制="只对所设生成机制成立；不模拟医院残差相关"))
        tg = pd.read_excel(T2X, sheet_name="G_MIX与POOLMIX")
        pm = tg[tg.选择器.astype(str).str.startswith("MIX − POOLMIX")]
        for cond in ("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H"):
            h = pm[pm.条件 == cond]
            if h.empty:
                continue
            h = h.iloc[0]
            rows.append(dict(编号=f"M-T2-MIXvsPOOLMIX-{cond}", 证据层级="独立 T2", 主张=f"{cond}：在所检验的融合协议下，引入专属预测的 MIX 与合并候选融合 POOLMIX 的差",
                             估计对象="两轨等权部署改善的同重复配对差（MIX − POOLMIX）",
                             关键数字=f"{100 * h.风险变化均值:.2f} 个百分点（SE {100 * h.SE:.2f}），MIX 更好 {int(h.MIX更好重复)}/{int(h.重复)}",
                             来源="T2 G 表", 限制="POOLMIX 只是 P* 与另一合并族候选的一种融合，不代表所有合并集成"))
        for a, b in (("N1", "N1-H"), ("P2", "P2-H")):
            ha = tc[(tc.条件 == a) & (tc.候选对 == "D|P*")]; hb = tc[(tc.条件 == b) & (tc.候选对 == "D|P*")]
            if ha.empty or hb.empty:
                continue
            txt = "；".join(f"{g} 轨 {a} {100 * ha[ha.轨 == g].iloc[0].真值相对收益均值:.2f}% vs {b} {100 * hb[hb.轨 == g].iloc[0].真值相对收益均值:.2f}%"
                           for g in ("A", "B"))
            rows.append(dict(编号=f"M-T2-重尾-{a}", 证据层级="独立 T2（分布压力条件）", 主张=f"{a} 的候选对真值收益在 t₅ 重尾协变量下的变化",
                             估计对象="HTFS_D vs P*_D 条件 Brier 真值相对收益（各条件全部完成重复的均值）", 关键数字=txt, 来源="T2 C 表",
                             限制="只改变协变量分布，结局函数与系数不变；不模拟医院残差相关"))
        for cond in ("N1", "N2", "N1-T", "N1-H"):
            f = tf[(tf.条件 == cond) & (tf.选择器 == "FCT|P*")]
            if f.empty:
                continue
            f = f.iloc[0]
            rows.append(dict(编号=f"M-T2-FCT-{cond}", 证据层级="独立 T2", 主张=f"{cond}：FCT|P* 的重复级「至少一轨不达标启用」",
                             估计对象=f"重复级比例（{int(f['≥1轨不达标启用_分母'])} 次）", 关键数字=f"{int(f['≥1轨不达标启用_分子'])}/{int(f['≥1轨不达标启用_分母'])}，CP 95% 上界 {100 * f['≥1轨不达标启用_CP上']:.2f}%",
                             来源="T2 F 表", 限制="有限重复的经验比例，不是错误率保证"))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = layers(); C = counts(L); MX, M = maxt(); F = fig_sources(L); X = matrix(L, C, MX)
    figs = pd.DataFrame([dict(图=k.split("_")[0], 表=k, 行数=len(v)) for k, v in F.items()])
    summ = dict(三层行数=L.groupby("层").size().to_dict(), maxT=MX.set_index("库")[["项数", "排除", "证据不足", "支持"]].to_dict("index"),
                矩阵行=len(X), 图源表=list(F))
    with pd.ExcelWriter(OUT / "R46_NUMBERS.xlsx", engine="openpyxl") as xw:
        L.to_excel(xw, sheet_name="N1_总结果表_三层", index=False)
        C.to_excel(xw, sheet_name="N2_分层计数", index=False)
        MX.to_excel(xw, sheet_name="N3_maxT汇总", index=False)
        X.to_excel(xw, sheet_name="N4_主张证据限制矩阵", index=False)
        figs.to_excel(xw, sheet_name="N5_主图源数据清单", index=False)
        for k, v in F.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    (OUT / "R46_NUMBERS.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return dict(L=L, C=C, MX=MX, M=M, F=F, X=X)


if __name__ == "__main__":
    main()
