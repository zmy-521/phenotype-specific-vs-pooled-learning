# -*- coding: utf-8 -*-
r"""R47 写作交接与图表索引（GPT 第 51 轮 §8）：md 与 xlsx **同源生成**——md 中每个数字都来自写入 xlsx「报告引用」表的同一 DataFrame。

不拟合、不模拟。输入 = pgss_r47_numbers.build()（只读 R41–R46 已交付结果表）；另读包测试结果 R47_PACKAGE_TESTS.json（若已存在）。
数字溯源两层：① 舍入匹配（pgss_r43_numaudit，打包自检时执行）；② 生成时逐值追踪：本脚本 pc / fx 与 numbers 模块 p2 格式化的每个原值，
必须在报告引用表中精确存在（相对差 ≤ 1e-12）；变异对照（原值 × (1 + 1e-6)）必须找不到；报告全文 SHA 登记在 R47_NUM_TRACE.json。
输出：29_R47_写作交接/R47_写作交接与图表索引.md、R47_冻结数字包.xlsx、R47_NUM_TRACE.json
用法：python pgss_r47_report.py
"""
import sys, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r47_numbers as N

OUT = N.OUT
MD = OUT / "R47_写作交接与图表索引.md"
TRACE = []
TRACE_MARK = "〔逐值追踪结果〕"
CITED = ["A1_报告层更正", "A2_输入与交付副本比对", "A3_T2逐重复记录导出", "A4_包测试", "B1_数据与任务概况", "B3_分层计数", "B4_maxT汇总",
         "B5_NHANES低白蛋白归因", "B6_NHANES差距收回", "C1_T2条件与设计", "C2_T2候选对真值收益", "C3_T2两轨等权部署", "C4_T2逐轨部署与变差",
         "C5_MIX等三面并列", "C5b_MIX与POOLMIX配对差", "C6b_T2重复级启用错误", "C6c_T2认证样本成本", "C7_T1与T2数值并列", "C8_两两排序反转_描述",
         "D1_主张证据限制矩阵", "D2_评价问题与证据", "D5_图索引", "D6_表格索引", "D7_复现入口"]


def pc(v, d=2):
    TRACE.append(float(v))
    return f"{100 * float(v):.{d}f}%"


def fx(v, d):
    TRACE.append(float(v))
    return f"{float(v):.{d}f}"


def esc(s):
    return str(s).replace("|", "\\|").replace("*", "\\*")


def tbl(df, cols, fmt=None, rename=None):
    fmt = fmt or {}; rename = rename or {}
    out = ["| " + " | ".join(esc(rename.get(c, c)) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("是" if v else "否")
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else str(int(v)) if float(v).is_integer() else f"{v:.6f}")
            else:
                cells.append(esc(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pick(df, **kw):
    return N.pick(df, **kw)


# ── 包测试（若已运行）──────────────────────────────────────────
def a4_tests():
    p = OUT / "R47_PACKAGE_TESTS.json"
    if not p.exists():
        return pd.DataFrame([dict(检查="包测试尚未运行", 通过=np.nan, 说明="")])
    t = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(t["检查"])


# ── md 各节 ─────────────────────────────────────────────────────
def sec0(S):
    a2, a3, a4 = S["A2_输入与交付副本比对"], S["A3_T2逐重复记录导出"], S["A4_包测试"]
    rec = a3[a3.表 != "rep.jsonl"]
    L = ["## 0. 先看这里\n",
         "**本轮性质**：按第 51 轮 §8 只做整理与写作支持。没有新拟合、没有新模拟、没有改动任何冻结目录。"
         "唯一的运行是合成示例：用冻结代码重新运行 T2 的一个正式重复，与交付记录逐位比较（不产生新数字），外加一次缩小网格的安装演示（数字不进入论文）。\n",
         f"**输入核对**：本包读取的 {len(a2)} 个结果表，与交接目录或交付 zip 内的副本逐字节相同 {int(a2.逐字节相同.sum())}/{len(a2)}（表 A2）。"
         f"T2 逐重复记录另附 CSV（%.17g 写出、round_trip 读回），与 parquet 逐位相同 {int(rec.CSV读回与parquet逐位相同.sum())}/{len(rec)}，"
         f"变异对照检出 {int(rec.变异对照检出.sum())}/{len(rec)}（表 A3）。这样没有 parquet 引擎的环境也能从逐重复记录独立重算 T2 均值。\n",
         "**三件交付**：本报告（索引与意见）；`R47_冻结数字包.xlsx`（全部表格、补充材料、方法定义与图源的唯一来源，首页「目录」）；"
         "`R47_CODE_AND_FIGURES.zip`（代码、六张主图 PNG/PDF、图源 CSV、T2 逐重复记录、合成示例、README）。\n"]
    if "检查" in a4.columns and a4.检查.iloc[0] != "包测试尚未运行":
        L.append(f"**包测试**：{int(a4.通过.sum())}/{len(a4)} 项通过（表 A4，见第 9 节）。\n")
    L.append("**我对第 51 轮的总体意见**：同意以 R46 结束核心计算、转入写作；同意四处报告层更正。GPT 引用的数字我逐一回到证据表核对，全部一致"
             "（例如 P2-H 的 FCT|P\\* 两轨等权区间、P2 A 轨 49/200、N1 MIX A 轨 64/200、P1 的 T1→T2 方向变化）。"
             "没有推翻性分歧；几条补充与一项待裁决的提议见第 10 节。\n")
    return "\n".join(L)


def sec1(S):
    A = S["A1_报告层更正"]
    L = ["## 1. 报告层更正（第 51 轮 §7、§8.2；只改报告与解释，不重训）\n",
         tbl(A, ["编号", "GPT要求", "R46原处", "R47做法", "依据表", "结果"]), ""]
    c3 = S["C3_T2两轨等权部署"]; f = c3[c3.选择器 == "FCT|P*"].set_index("条件").loc[N.CONDS].reset_index()
    L.append("FCT|P\\* 两轨等权部署改善逐条件（表 C3；区间 = 200 次重复均值 ± 1.96·SE）：\n")
    L.append(tbl(f, ["条件", "两轨等权部署改善均值", "下", "上", "均值±1.96SE方向"],
                 fmt={"两轨等权部署改善均值": pc, "下": pc, "上": pc}))
    c7 = S["C7_T1与T2数值并列"]; cg = c7[c7.T1与T2 == "T2 区间支持的方向与 T1 均值异号"]
    vc = c7.T1与T2.value_counts()
    L.append(f"\nT1 与 T2 逐项并列（表 C7，共 {len(c7)} 行）：" + "；".join(f"{k} {v}" for k, v in vc.items()) + "。方向改变的行：\n")
    L.append(tbl(cg, ["条件", "对象", "轨", "T1均值", "T2均值", "T2下", "T2上"], fmt={"T1均值": pc, "T2均值": pc, "T2下": pc, "T2上": pc}))
    L.append("\n其余更正的落点：极端值与尾部比例只来自每次重复最初的 10 万评价样本（表 C9 列名已改）；两两排序反转标为描述性（表 C8）；"
             "FCT 的启用错误标为指定条件下的 Monte Carlo 估计（表 C6a、C6b）；MIX 的平均、逐轨变差与变异并列（表 C5）。\n")
    return "\n".join(L)


def sec2(S):
    B3, B4, B5, B6 = S["B3_分层计数"], S["B4_maxT汇总"], S["B5_NHANES低白蛋白归因"], S["B6_NHANES差距收回"]
    L = ["## 2. 冻结数字\n", "### 2.1 真实数据三层（不合并成一个总效应）\n",
         "L1 = ICU 预定比较（MIMIC-IV、eICU-CRD）与同一程序的独立复现（NWICU）；L2 = NHANES 原始跨领域复现；L3 = NHANES 结果知情后登记的目标一致性追加对照。"
         "区间为固定 OOF 预测的联合簇重抽 95% 区间；三态以 m = 1% 判读。\n"]
    b = B3.copy()
    L.append(tbl(b, ["层", "库", "对照", "尺度", "任务", "排除", "证据不足", "支持", "全负", "跨零", "全正", "相对改善中位", "相对上界最大"],
                 fmt={"相对改善中位": pc, "相对上界最大": pc}))
    L.append("\n逐轨 max-T 同时判断（表 B4）：\n")
    L.append(tbl(B4, ["库", "项数", "排除", "证据不足", "支持", "max_T临界值", "证据不足项"], fmt={"max_T临界值": lambda v: fx(v, 3)}))
    k = B5[B5.与目标一致的尺度 | ((B5.对照 == "E1") & B5.尺度.isin(["B", "pop"]))]
    L.append("\nNHANES 低白蛋白（10 年全因死亡）：原比较与目标一致对照（表 B5，全部对照与尺度见该表）：\n")
    L.append(tbl(k, ["证据层", "对照", "尺度", "观察相对改善", "相对_P2_5", "相对_P97_5", "三态_m1"],
                 fmt={"观察相对改善": pc, "相对_P2_5": pc, "相对_P97_5": pc}))
    L.append("\n绝对 Brier（表 B6）：\n")
    L.append(tbl(B6, ["目标", "尺度", "R_P*（两轨等权开发）", "R_P*_目标", "R_HTFS_目标", "R_HTFS（原）"],
                 fmt={c: (lambda v: fx(v, 6)) for c in ("R_P*（两轨等权开发）", "R_P*_目标", "R_HTFS_目标", "R_HTFS（原）")}))
    C2, C3, C5, C6b, C6c = S["C2_T2候选对真值收益"], S["C3_T2两轨等权部署"], S["C5_MIX等三面并列"], S["C6b_T2重复级启用错误"], S["C6c_T2认证样本成本"]
    L.append("\n### 2.2 T2（8 条件 × 200 次；n = 6,000、π_A = 0.15；条件 Brier 真值）\n")
    p = C2[C2.候选对 == "D|P*"]
    w = p.pivot_table(index="条件", columns="轨", values=["真值相对收益均值", "SE"]).loc[N.CONDS]
    w.columns = [f"{a}_{b}" for a, b in w.columns]; w = w.reset_index()
    dirs = p.set_index(["条件", "轨"]).相对m方向
    w["A轨相对m"] = [dirs[(c, "A")] for c in w.条件]; w["B轨相对m"] = [dirs[(c, "B")] for c in w.条件]
    L.append("HTFS_D 相对 P\\*_D 的真值相对收益（表 C2；均值、SE 为 200 次重复）：\n")
    L.append(tbl(w, ["条件", "真值相对收益均值_A", "SE_A", "A轨相对m", "真值相对收益均值_B", "SE_B", "B轨相对m"],
                 fmt={"真值相对收益均值_A": pc, "SE_A": pc, "真值相对收益均值_B": pc, "SE_B": pc},
                 rename={"真值相对收益均值_A": "A 轨均值", "SE_A": "A 轨 SE", "真值相对收益均值_B": "B 轨均值", "SE_B": "B 轨 SE"}))
    sel = ["ALWAYS_SPLIT", "CV_POINT_M|P*", "FCT|P*", "MIX", "POOLMIX"]
    q = C3[C3.选择器.isin(sel)].pivot_table(index="条件", columns="选择器", values="两轨等权部署改善均值").loc[N.CONDS, sel].reset_index()
    L.append("\n选择器两轨等权部署改善（相对 P\\*_D；表 C3 另有 SE、区间与方向，全部 13 个选择器）：\n")
    L.append(tbl(q, ["条件"] + sel, fmt={s: pc for s in sel}))
    m = C5[C5.选择器 == "MIX"]
    L.append("\nMIX 三面并列（表 C5；实际变差 = 部署改善 < 0 且 |改善| ≥ 2 个积分 SE）：\n")
    L.append(tbl(m, ["条件", "两轨等权均值", "两轨等权方向", "两轨等权P5", "两轨等权P95", "A轨均值", "A轨方向", "A轨实际变差", "B轨均值", "B轨实际变差"],
                 fmt={"两轨等权均值": pc, "两轨等权P5": pc, "两轨等权P95": pc, "A轨均值": pc, "B轨均值": pc},
                 rename={"A轨实际变差": "A轨实际变差/200", "B轨实际变差": "B轨实际变差/200"}))
    f = C6b[C6b.选择器 == "FCT|P*"].set_index("条件").loc[N.CONDS].reset_index()
    L.append("\nFCT|P\\* 重复级「至少一轨不达标启用」（表 C6b；指定生成条件下 200 次重复的 Monte Carlo 估计，不是错误率保证）：\n")
    L.append(tbl(f, ["条件", "≥1轨不达标启用_分子", "≥1轨不达标启用_分母", "≥1轨不达标启用_CP上"], fmt={"≥1轨不达标启用_CP上": pc}))
    L.append("\n认证样本成本 P\\*_F 相对 P\\*_D（表 C6c）：\n")
    L.append(tbl(C6c, ["条件", "轨", "P*_F相对P*_D改善均值", "SE", "均值±1.96SE方向"], fmt={"P*_F相对P*_D改善均值": pc, "SE": pc}))
    L.append("")
    return "\n".join(L)


def sec3(S):
    D1 = S["D1_主张证据限制矩阵"]
    L = ["## 3. 主张—证据—限制矩阵（表 D1）\n",
         "「主张」取自第 51 轮原文的最窄表述并注明节号（K10 取自 R46 表 D6），关键数字由脚本从本包的表取出；我没有另写带方向的结论。\n",
         tbl(D1, ["编号", "出处", "证据层级", "主张", "关键数字", "限制", "不应写成"]), ""]
    return "\n".join(L)


def sec4(S):
    D2 = S["D2_评价问题与证据"]
    return "\n".join(["## 4. 评价顺序与本研究中的证据（表 D2；第 51 轮 §四）\n",
                      tbl(D2, ["顺序", "必须回答的问题", "本研究中的估计量", "证据", "证据性质"]), ""])


def sec5(S):
    D5 = S["D5_图索引"]
    L = ["## 5. 六张主图（英文标注；每个点来自 xlsx 中 F 开头的图源表，图源也以 CSV 放在 zip 的 figure_sources/）\n",
         tbl(D5, ["图", "文件", "内容", "源表", "源表行数"]), "",
         "说明：图只呈现数据，不带结论性标题。Fig 1 是示意图，其中的任务数、条件数与重复数来自 F1_研究设计。Fig 4 只画全部外层折可评价的点"
         "（eICU-CRD 在 25% 与 50% 没有这样的点）；学习曲线只在 MIMIC-IV 与 eICU-CRD 运行过。图题、图注由 GPT 撰写；改成中文或调整版式只需改 `pgss_r47_figures.py`。\n"]
    return "\n".join(L)


def sec6(S):
    return "\n".join(["## 6. 表格索引（建议位置，GPT 定）\n", tbl(S["D6_表格索引"], ["建议位置", "表", "内容"]), "",
                      "主表 1 的规模来源：MIMIC-IV、eICU-CRD 取 R41 任务与轨表（data v5）；NWICU 取 R46 修正后推断（住院死亡已排除无生命状态证据的人次）；"
                      "NHANES 取 R46 修正后推断（O_CVD10×低白蛋白不可执行，列出但标未纳入）。\n"])


def sec7(S):
    D3 = S["D3_方法定义"]
    return "\n".join(["## 7. 方法定义（表 D3，唯一来源）\n",
                      f"共 {len(D3)} 条，按类别：" + "；".join(f"{k} {v}" for k, v in D3.类别.value_counts().items()) + "。"
                      "引擎网格、早停与校准设置、T2 常量与结局函数源码在生成时直接从冻结模块读取（不是手抄），因此与实际运行的程序一致。\n"])


def sec8(S):
    D4 = S["D4_文献比较"]
    th = ["信息提供", "架构比较", "目标一致性", "选择成本"]
    cov = D4[["编号"] + th].copy()
    for t in th:
        cov[t] = cov[t].map(lambda s: "未摘录" if s.startswith("本表未摘录") else "有原文摘录")
    return "\n".join(["## 8. 直接相关原始研究的比较表（表 D4）\n",
                      "只用 R46 已在本地原文全文中逐字核对的摘录（11 篇，每条注明页码）；没有摘录的格子写「本表未摘录（未核对）」，不推测。"
                      "Pfohl 2022（合并训练与按亚组单独训练比较；按亚组最坏目标改早停与模型选择）、Ustun 2019、Suriyakumar 2023 均在表中，并写明各自的估计目标与参照。\n",
                      tbl(cov, ["编号"] + th), ""])


def sec9(S):
    D7, A4 = S["D7_复现入口"], S["A4_包测试"]
    L = ["## 9. 代码包与合成示例\n",
         "zip 结构：`code/`（本轮与全部既往脚本的 import 闭包；冻结脚本一字未改）、`figures/`、`figure_sources/`（图源 CSV）、`t2_records/`（T2 逐重复记录 parquet + CSV）、"
         "`README.md`（入口与目录布局）；数字包 xlsx 单独交付，不在 zip 内重复。\n",
         "**合成示例**（无需受限数据）：`python code/pgss_r47_example.py reproduce` 用冻结代码重新运行 T2 的 P2 第 0 次重复，并与 `t2_records/` 逐位比较；"
         "`quick` 为缩小网格的安装演示（约 2 分钟，数字不可与论文比较）。\n",
         "**历史目录依赖**：冻结脚本（引擎、各轮运行器与推断）保持原样——改路径会改变 GPT 已核对过的代码指纹。新入口 `pgss_r47_example.py` 与路径无关；"
         "真实数据流水线的目录布局与入口见表 D7 与 README。若需要把全部脚本改成与路径无关，需要另做一轮等价验证，请 GPT 定。\n",
         tbl(D7, ["步骤", "脚本", "输入", "输出"]), "", "包测试（表 A4；每项的说明、耗时与检出层见该表）：\n", tbl(A4, ["检查", "通过"]), ""]
    return "\n".join(L)


def sec10(S):
    C4 = S["C4_T2逐轨部署与变差"]

    def fw(g):
        h = C4[(C4.选择器 == "FCT|P*") & (C4.轨 == g) & C4.条件.isin(["N1", "N2", "N1-T", "N1-H"])]
        return f"{int(h.部署实际变差_分子.min())}–{int(h.部署实际变差_分子.max())}/{int(h.部署实际变差_分母.iloc[0])}"
    L = ["## 10. 意见（请 GPT 裁决或知悉）\n",
         "1. **同意第 51 轮的总体判断与四处更正**，更正已全部落在数字包（第 1 节）。另外一处措辞我建议同样处理：R46 报告 §0 写的「Q1–Q5 一致 14、推翻 0」"
         "是相对方案文字的机械归类；论文里如需引用 T1，应引用表 C7 的数值并列，不引用这组计数。",
         "2. **复核边界**：第 51 轮说明本地缺 parquet 引擎、没有从逐重复记录重算 T2 均值。本包在 `t2_records/` 附了逐位保真的 CSV（表 A3），"
         "可以直接从 1,600 条记录独立重算表 C2–C6 的任意数字。我在 R46 交付后用同一批 parquet 重算过 40 个数字并全部吻合，但那仍是我自己核自己。",
         "3. **合成示例设计为「逐位重现冻结重复」而不是新模拟**：这样示例既能证明代码包可用，又不产生任何新的研究数字，符合「不新增模拟」。",
         "4. **一项提议，不执行**：第 51 轮把「存在收益—能识别—最终获益」作为主线。T2 已有的逐重复记录可以只读地算出每个条件的「逐轨真值择优」"
         "（每轨在 P\\*_D 与 HTFS_D 中取真值风险较低者）的两轨等权改善，作为选择程序能达到的上限，与各选择器的实际部署改善相减，得到「选择差距」。"
         "它不需要任何新拟合或新模拟，但属于新的派生分析；第 51 轮写明「不因结果不够整齐而继续分析」，所以我只提出，不计算。要不要做由你定。",
         "5. **图用英文标注**（投稿期刊为英文）；图题与图注留给正文。Fig 6b 中 FCT|P\\* 的逐轨实际变差比例在无轨道特异结构的 4 个条件里为 "
         f"A 轨 {fw('A')}、B 轨 {fw('B')}（表 C4）。FCT 不启用时部署的是只用 2/3 数据拟合的 P\\*_F，所以这组数与表 C6c 的样本成本是同一件事的两种呈现；"
         "图注需要说明，否则读者会把它读成启用错误。",
         "6. **K10（N2 中 MIX − POOLMIX 的配对差）** 第 51 轮没有讨论，数字见表 C5b；是否写入正文由你定。", ""]
    return "\n".join(L)


def trace_check(sheets):
    arr = [pd.DataFrame(v).apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel() for k, v in sheets.items() if k in CITED]
    pool = np.unique(np.abs(np.concatenate(arr))); pool = pool[np.isfinite(pool)]

    def found(x):
        x = abs(x); i = int(np.searchsorted(pool, x)); c = pool[max(i - 1, 0):i + 1]
        return bool(len(c) and np.min(np.abs(c - x)) <= 1e-12 * max(1.0, x))
    allv = TRACE + N.TRACE
    tv = np.unique(np.abs(np.array(allv, float)))
    miss = [float(x) for x in tv if not found(x)]
    nz = tv[tv > 0]
    pick_ = np.random.default_rng(0).choice(nz, size=min(50, len(nz)), replace=False) if len(nz) else np.array([])
    leak = int(sum(found(x * (1 + 1e-6)) for x in pick_))
    return dict(追踪数值=len(allv), 不同数值=int(len(tv)), 找不到=miss, 变异对照数=int(len(pick_)), 变异对照被放行=leak)


def main():
    TRACE.clear()
    S = N.build()
    S["A4_包测试"] = a4_tests()
    sh = N.sheets(S)
    for k in CITED:
        assert k in sh, k
    assert all(len(k) <= 31 for k in sh), [k for k in sh if len(k) > 31]
    L = ["# R47 写作交接与图表索引\n",
         "> 由 `pgss_r47_report.py` 生成；正文数字只来自同名数字包中「目录」标为「报告引用」的表（md 与 xlsx 同源）。相对改善 = (R参照 − R候选)/R参照，"
         "正值 = 候选更好；三态以 m = 1% 判读。真实数据区间为固定 OOF 预测的簇重抽；T2 区间为 200 次重复的均值 ± 1.96·SE。\n"]
    for f in (sec0, sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec10):
        L.append(f(S))
    L.append("## 11. 数字溯源\n\n" + TRACE_MARK + "。\n")
    tc = trace_check(sh)
    txt = "\n".join(L) + "\n"
    assert txt.count(TRACE_MARK) == 1
    txt = txt.replace(TRACE_MARK, f"生成时逐值追踪 {tc['追踪数值']} 次格式化（{tc['不同数值']} 个不同原值），在报告引用表中找不到原值的 {len(tc['找不到'])} 个；"
                                  f"变异对照（{tc['变异对照数']} 个原值乘以 1 + 10⁻⁶）被放行 {tc['变异对照被放行']} 个；"
                                  "追踪记录同时登记报告全文的 SHA-256，打包自检据此检出生成后对报告的任何改动（`R47_NUM_TRACE.json`）")
    note = ["R47 冻结数字包：只读 R41–R46 已交付结果表（输入逐字节核对见 A2），不拟合、不模拟。",
            "表名首字母：A 更正与核对；B 真实数据三层；C T2 与模拟；D 写作资产（主张矩阵、评价顺序、方法定义、文献、图表索引、复现入口）；F 图源（图中每个点）。",
            "相对改善 = (R参照 − R候选)/R参照，正值 = 候选更好；m = 1%。真实数据区间为固定 OOF 预测的联合簇重抽；T2 区间为 200 次重复均值 ± 1.96·SE。",
            "「主张」列取自 GPT 第 51 轮原文并注明出处；Claude 不另写带方向的结论。"]
    N.write_xlsx(S, CITED, note=note)
    MD.write_text(txt, encoding="utf-8")
    tc["报告sha"] = hashlib.sha256(MD.read_bytes()).hexdigest()[:16]
    (OUT / "R47_NUM_TRACE.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"md {len(txt)} 字；追踪 {tc['追踪数值']}（{tc['不同数值']} 不同），找不到 {len(tc['找不到'])}，变异放行 {tc['变异对照被放行']}")
    if tc["找不到"]:
        print("找不到：", tc["找不到"][:20])
    return tc


if __name__ == "__main__":
    tc = main()
    sys.exit(0 if not tc["找不到"] and not tc["变异对照被放行"] else 1)
