# -*- coding: utf-8 -*-
r"""R49 只读补齐报告（GPT 第 53 轮）：md 与 xlsx 同源生成。只写两项资料、来源、核验与 R48 索引的增量建议，不写科研裁决、不写正文。

输入 = pgss_r49_cohort.build()（任务一）+ pgss_r49_aux.build()（任务二）；另读包测试结果 R49_PACKAGE_TESTS.json（若已存在）。
数字溯源：fx 格式化的每个原值必须在报告引用表中精确存在；变异对照（× (1 + 1e-6)）必须找不到；报告 SHA 登记在 R49_NUM_TRACE.json。
输出：31_R49_只读补齐/R49_只读补齐_报告.md、R49_只读补齐_证据表.xlsx、R49_NUM_TRACE.json
"""
import sys, json, pathlib, hashlib, re
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r49_cohort as CO
import pgss_r49_aux as AX

ROOT = HERE.parent
OUT = ROOT / "31_R49_只读补齐"
MD = OUT / "R49_只读补齐_报告.md"
XL = OUT / "R49_只读补齐_证据表.xlsx"
R48XL = ROOT / "30_R48_图稿与补充材料" / "R48_图源与补充材料.xlsx"
TRACE = []
TRACE_MARK = "〔逐值追踪结果〕"
CITED = ["G1_缺口关闭判定", "G2_新表清单", "G3_R48索引增量建议", "A1_来源与版本", "A2_v5与R35队列逐条比较", "A3_R35运行记录与交付副本",
         "A4_任务二输入与交付副本", "A5_R43格式验证", "A6_与R46_X1交叉核对", "A7_包测试", "C1_纳排流程", "C1b_各步排除原因", "C2_表型任务", "C4_计数口径",
         "D0_版本判定", "D3_配对差汇总_R43格式", "D4_校准诊断汇总_R43格式"]
CAT = {"G": "结论与索引建议", "A": "来源与核验", "C": "任务一：MIMIC-IV / eICU-CRD 纳排流程", "D": "任务二：辅助指标跨 seed 汇总"}


def fx(v, d=6):
    TRACE.append(float(v))
    return f"{float(v):.{d}f}"


def esc(s):
    return str(s).replace("|", "\\|").replace("*", "\\*")


def tbl(df, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and not (isinstance(v, float) and not np.isfinite(v)):
                cells.append(fmt[c](v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("是" if v else "否")
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else str(int(v)) if float(v).is_integer() else esc(v))
            elif v is None:
                cells.append("—")
            else:
                cells.append(esc(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def a7_tests():
    p = OUT / "R49_PACKAGE_TESTS.json"
    if not p.exists():
        return pd.DataFrame([dict(检查="包测试尚未运行", 通过=np.nan, 说明="")])
    return pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))["检查"])


def g1_gaps(S):
    c1, c2, a2, a3 = S["C1_纳排流程"], S["C2_表型任务"], S["A2_v5与R35队列逐条比较"], S["A3_R35运行记录与交付副本"]
    d0, a5, a6 = S["D0_版本判定"], S["A5_R43格式验证"], S["A6_与R46_X1交叉核对"]
    rec_only = c1[c1.来源.str.startswith("仅")]
    ok1 = bool(a2.一致.all() and a3.逐值相同.all() and c2[["与R41一致", "与R42评价单位一致", "与R47一致"]].all().all())
    ok2 = bool(a5.一致.all() and a6.通过.all())
    return pd.DataFrame([
        dict(缺口="MIMIC-IV/eICU-CRD 终版纳排流程（R48 S0 组 1）", 判定="关闭" if ok1 else "未关闭",
             依据=f"data v5 的合格标志与两个结局和 R35 冻结队列逐条相同；R35 运行记录与 R35 交付工作簿逐值相同；{len(c2)} 个任务的人数、事件与 A/B 轨计数与 R41、R42、R47 全部一致",
             保留说明=f"两库各 {len(rec_only) // 2} 个中间步骤（landmark 时仍在 ICU、成人）只有 R35 运行记录，冻结数据没有保存该步标志，无法独立重算；eICU 最后一步的两项排除原因无法拆分（不推算）"),
        dict(缺口="NWICU/NHANES 辅助指标跨 seed 汇总（R48 S0 组 6）", 判定="关闭（原始分析）" if ok2 else "未关闭",
             依据=f"逐任务×模型×范围的三个 seed 均值与范围、配对差均值与范围、R43 格式汇总均已给出；汇总函数从 R43 B2/B1 重算 B3/B4 一致；与 R46 X1 独立交叉核对四库全部通过",
             保留说明="NHANES 目标一致性追加臂在冻结结果中只有校准与 Brier（R46 X1）及 OOF 的 Brier/logloss 对照，没有逐 seed 的 AUROC、AUPRC；本轮不补算"),
    ])


def g2_new(S):
    rows = [("C1_纳排流程", "任务一", "R35 运行记录（与交付工作簿逐值相同）+ data v5 META 重算", "data v5（DATA_VERSION.json v5_SHA 一致）", "A1、A2、A3；可重算的步骤与运行记录一致"),
            ("C1b_各步排除原因", "任务一", "R35 运行记录与未知/冲突登记", "R35（与 data v5 队列逐条相同）", "分解只写记录给出或逻辑推出的；其余标「不可恢复」"),
            ("C1c_R35未知与冲突登记", "任务一", "R35 运行记录原样", "R35", "A3（与交付副本逐值相同）"),
            ("C2_表型任务", "任务一", "data v5 META × PHENO × PHENOOK 计数", "data v5", "与 R41 A1、R42 任务完整度、R47 B1 逐项一致"),
            ("C3_定义与landmark", "任务一", "pgss_r35_cohort.py、pgss_r35_qc.phenotypes 源码；DATA_VERSION.json", "冻结代码（SHA 与 R35 交付 / R47 登记一致）", "A1"),
            ("C4_计数口径", "任务一", "data v5 META + R35 队列 parquet", "data v5", "住院键与分析单位一一对应"),
            ("D0_版本判定", "任务二", "v2 与 v1 逐 seed 比较", "NWICU v2、NHANES v2（R46）", "重跑任务集合与 R46 记录一致（包测试）"),
            ("D1_各模型跨seed汇总", "任务二", "逐 seed 辅助指标（ICU：R43 B1；NWICU、NHANES：v2）", "见 D0", "A6（与 R46 X1 交叉核对）；包测试独立复算"),
            ("D2_配对差跨seed汇总", "任务二", "逐 seed 配对差（ICU：R43 B2；NWICU、NHANES：v2）", "见 D0", "包测试独立复算"),
            ("D3_配对差汇总_R43格式", "任务二", "同一汇总函数（R43 定义）", "见 D0", "A5（从 R43 B2 重算 B3 一致）"),
            ("D4_校准诊断汇总_R43格式", "任务二", "同一汇总函数（R43 定义）", "见 D0", "A5（从 R43 B1 重算 B4 一致）"),
            ("D5_NHANES目标一致性追加臂", "任务二", "R46 证据表 X1 原样", "NHANES 目标一致性 v2", "原样复制（包测试逐值比较）")]
    d = pd.DataFrame(rows, columns=["表", "任务", "来源", "版本", "数字核验"])
    d["行数"] = [len(S[k]) for k in d.表]
    return d


def g3_index(S):
    s0 = pd.read_excel(R48XL, sheet_name="S0_补充材料索引")
    mx = max(int(x[1:]) for x in s0.建议编号.dropna() if isinstance(x, str) and x.startswith("S"))
    rows = [("1 数据、队列与任务", "MIMIC-IV/eICU-CRD 纳排流程（data v5 终版口径）", "缺口 → 本轮补齐", "C1_纳排流程；C1b_各步排除原因；C1c_R35未知与冲突登记"),
            ("1 数据、队列与任务", "（新增）ICU 六个表型的任务规模、患者数与表型不可判定数", "新增", "C2_表型任务"),
            ("1 数据、队列与任务", "（新增）表型定义、landmark 规则与计数口径", "新增", "C3_定义与landmark；C4_计数口径"),
            ("6 辅助性能（AUROC、AUPRC、logloss、校准截距与斜率）", "NWICU、NHANES 辅助指标跨 seed 汇总", "缺口 → 本轮补齐", "D1_各模型跨seed汇总；D2_配对差跨seed汇总"),
            ("6 辅助性能（AUROC、AUPRC、logloss、校准截距与斜率）", "（新增）四库 R43 格式汇总", "新增", "D3_配对差汇总_R43格式；D4_校准诊断汇总_R43格式"),
            ("5 NHANES 目标一致性追加对照（v2）", "（说明）追加臂的辅助指标可得范围", "仅索引（R48 S51 的子集）", "D5_NHANES目标一致性追加臂")]
    d = pd.DataFrame(rows, columns=["R48组", "条目", "建议处理", "R49表"])
    nums, k = [], mx
    for h in d.建议处理:
        if h.startswith("仅索引"):
            nums.append("—")
        else:
            k += 1; nums.append(f"S{k}")
    d.insert(0, "建议编号（接在 R48 之后，不重编既有编号）", nums)
    d["R48当前最大编号"] = f"S{mx}"
    return d


def write_xlsx(S, path=XL):
    sh = {k: v for k, v in S.items() if isinstance(v, pd.DataFrame)}
    def key(k):
        m = re.match(r"([A-Z])(\d+)", k)
        return ("GACD".index(k[0]), int(m.group(2)) if m else 0, k)
    order = sorted(sh, key=key)
    toc = pd.DataFrame([dict(表=k, 类别=CAT[k[0]], 行数=len(sh[k]), 报告引用="是" if k in CITED else "否") for k in order])
    note = ["R49（GPT 第 53 轮）：两项只读资料补齐。不拟合、不模拟、不重新校准、不修改 data v5、不使用 R34 的任何人数。",
            "任务一（C 表）：MIMIC-IV / eICU-CRD 终版纳排流程。分析单位 = 每次住院的一段 ICU 住院；患者数按 data v5 META.group_id（MIMIC subject_id、eICU uniquepid）。",
            "任务二（D 表）：三个 seed 的描述性汇总，只报均值与最小–最大范围；seed 不是独立样本，不报跨 seed 的标准误或区间。",
            "pop = 研究样本比例加权（按记录等权），不是 NHANES 调查权重；NHANES 目标一致性追加臂单列在 D5。",
            "G3 只提出 R48 补充材料索引的增量建议，不重编既有编号。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        toc.to_excel(xw, sheet_name="目录", index=False)
        pd.DataFrame({"说明": note}).to_excel(xw, sheet_name="0_说明", index=False)
        for k in order:
            sh[k].to_excel(xw, sheet_name=k[:31], index=False)
    return toc


def trace_check(sheets):
    pool = np.concatenate([sheets[k].apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel() for k in CITED if k in sheets])
    pool = pool[np.isfinite(pool)]
    found = lambda x: bool(np.any(np.abs(pool - x) <= 1e-12 * max(1.0, abs(x))))
    miss = sorted({x for x in TRACE if not found(x)})
    mut = [x * (1 + 1e-6) for x in sorted(set(TRACE)) if x != 0][:50]
    return dict(追踪数值=len(TRACE), 不同数值=len(set(TRACE)), 找不到=miss, 变异对照数=len(mut), 变异对照被放行=int(sum(found(x) for x in mut)))


def main():
    TRACE.clear()
    S = {}
    S.update(CO.build()); S.update(AX.build())
    S["A7_包测试"] = a7_tests()
    S["G1_缺口关闭判定"] = g1_gaps(S)
    S["G2_新表清单"] = g2_new(S)
    S["G3_R48索引增量建议"] = g3_index(S)
    c1, c1b, c2, c4 = S["C1_纳排流程"], S["C1b_各步排除原因"], S["C2_表型任务"], S["C4_计数口径"]
    d0, d3, d4 = S["D0_版本判定"], S["D3_配对差汇总_R43格式"], S["D4_校准诊断汇总_R43格式"]
    a1, a2, a3, a4, a5, a6, a7 = (S[k] for k in ("A1_来源与版本", "A2_v5与R35队列逐条比较", "A3_R35运行记录与交付副本", "A4_任务二输入与交付副本",
                                                 "A5_R43格式验证", "A6_与R46_X1交叉核对", "A7_包测试"))
    npass = int(sum(x is True or x == 1 for x in a7.通过))
    rer = d0[(d0.表 == "6_辅助指标逐seed") & ~d0.与v1逐值相同]
    L = ["# R49 只读补齐报告\n",
         "> 由 `pgss_r49_report.py` 生成；md 与 `R49_只读补齐_证据表.xlsx` 同源。本轮只补两项资料：不拟合、不模拟、不重新校准、不修改 data v5，不使用 R34 的任何人数，不写正文。\n",
         "## 0. 两个缺口是否关闭\n", tbl(S["G1_缺口关闭判定"], ["缺口", "判定", "依据", "保留说明"]) + "\n",
         f"**输入核对**：任务一的 data v5 文件与 DATA_VERSION.json 登记的 SHA 一致，队列定义脚本与 R35 交付代码逐字节相同（表 A1）；任务二的 {len(a4)} 个输入与交付副本逐字节相同 "
         f"{int(a4.逐字节相同.sum())}/{len(a4)}（表 A4）。**包测试** {npass}/{len(a7)} 项通过（表 A7）。\n",
         "## 1. 任务一：MIMIC-IV / eICU-CRD 终版纳排流程\n",
         "**来源链**：队列与结局只由 `pgss_r35_cohort.py` 定义；data v5 的 META 沿用这一队列（v2–v5 只改预测变量与表型输入，不改队列）。"
         "本轮逐条比较 data v5 META 与 R35 冻结队列（表 A2）："
         + "；".join(f"{r.数据库} {r.字段} 不同 {r.不同的单位}/{r.分析单位}" for r in a2.itertuples()) + "。因此 R35 的流程就是 data v5 的流程。"
         "R35 运行记录与 R35 当轮交付的工作簿逐值相同（表 A3）。\n",
         "**标准流程表**（表 C1；住院单位 = 每次住院的一段 ICU 住院；患者数按 data v5 的 group_id）：\n",
         tbl(c1, ["数据库", "步骤", "步骤说明", "Step", "住院单位数", "排除", "患者数", "重算与运行记录一致", "来源", "事件"]) + "\n",
         "**各步排除原因**（表 C1b；只写记录直接给出或由记录逻辑推出的，其余写不可恢复，不推算）：\n",
         tbl(c1b, ["数据库", "步骤", "排除", "原因", "分解", "依据"]) + "\n",
         "**计数口径**（表 C4）：\n", tbl(c4, ["数据库", "分析单位", "住院键", "患者键", "分析单位数", "患者数", "合格单位数", "合格患者数", "合格医院数", "推断的重抽单位"]) + "\n",
         "**六个表型的最终任务**（表 C2；任务 = 合格队列中该表型可判定者；与 R41 A1、R42 任务完整度、R47 B1 逐项对账）：\n",
         tbl(c2, ["数据库", "结局", "表型", "表型不可判定", "任务n", "任务患者数", "事件", "A轨n", "A轨事件", "B轨n", "B轨事件", "与R41一致", "与R42评价单位一致", "与R47一致"]) + "\n",
         "表型定义直接取自 `pgss_r35_qc.phenotypes` 源码（v5 构建脚本调用的同一函数），landmark 规则与结局定义取自 `pgss_r35_cohort.py` 与 DATA_VERSION.json，见表 C3。\n",
         "## 2. 任务二：NWICU / NHANES 辅助指标跨 seed 汇总\n",
         "**版本**（表 D0，逐任务把 v2 与 v1 的逐 seed 记录全部比较）：R46 重跑、以 v2 为有效版本的任务：" +
         "；".join(f"{db} {oc}：" + "、".join(h.表型) for (db, oc), h in rer.groupby(["数据库", "结局"])) +
         "。其余任务与 v1 逐值相同，即沿用既有有效结果。\n",
         "**逐任务表**：表 D1 给出每个任务 × 模型（P\\*、HTFS、Hfull、S0\\*、SZ\\*）× 范围（全体、A、B、两轨等权）的 AUROC、AUPRC、Brier、logloss、校准截距、校准斜率，"
         "各为三个 seed 的均值、最小、最大；表 D2 给出各候选相对参照的配对差（同样是均值、最小、最大）。为统一格式，MIMIC-IV 与 eICU-CRD 也按同一方式从 R43 B1/B2 汇总。"
         "seed 不是独立样本，表中不报跨 seed 的标准误或区间。\n",
         "**R43 格式汇总**（表 D3，HTFS 相对 P\\*；差 = HTFS − P\\*，Brier/logloss 差 > 0 表示 HTFS 更差，AUROC/AUPRC 差 > 0 表示 HTFS 更好）：\n"]
    h = d3[(d3.候选 == "HTFS") & (d3.参照 == "P*")]
    f6 = {c: (lambda v: fx(v)) for c in ("Brier差中位", "logloss差中位", "AUROC差中位", "AUPRC差中位", "校准截距差中位", "校准斜率差中位")}
    L.append(tbl(h, ["数据库", "范围", "任务乘seed", "Brier差中位", "Brier差为正", "logloss差中位", "logloss差为正", "AUROC差中位", "AUROC差为正", "AUPRC差中位", "AUPRC差为正",
                     "校准斜率差中位", "校准斜率差为正"], fmt=f6) + "\n")
    L.append("全部候选—参照对见表 D3；各模型 A、B 两轨校准截距与斜率的分布见表 D4（与 R43 B4 同一定义）。\n")
    L.append("**NHANES 目标一致性追加臂**（表 D5）：结果知情后登记的追加对照，与原始分析分开。冻结结果中只有各臂的校准截距、斜率与 Brier"
             "（R46 X1，跨 seed 均值与范围，原样列出），以及 OOF 的 Brier、logloss 对照（R48 的 S37、S38）；逐 seed 的 AUROC、AUPRC 当时没有计算，本轮不补算。"
             "pop 是研究样本比例加权，不是 NHANES 调查权重。\n")
    L += ["## 3. 核验\n",
          tbl(a1, ["对象", "本地SHA16", "对照", "对照SHA16", "一致"]) + "\n",
          tbl(a3, ["记录", "交付副本", "行数", "逐值相同"]) + "\n",
          tbl(a4, ["输入", "交付副本", "逐字节相同"]) + "\n",
          "汇总函数的格式验证（表 A5）：" + "；".join(f"{r.检查}：{'一致' if r.一致 else '不一致'}（{r.容差}）" for r in a5.itertuples()) + "。\n",
          "与 R46 X1 的独立交叉核对（表 A6；X1 由另一份代码从预测直接计算跨 seed 的校准均值、最小、最大与 Brier）：" +
          "；".join(f"{r.数据库} 匹配 {r.匹配行}/{r.本表可比行} 行，{'通过' if r.通过 else '未通过'}" for r in a6.itertuples()) + "。\n",
          "## 4. R48 补充材料索引的增量更新建议（表 G3；只增不改，既有编号不动）\n",
          tbl(S["G3_R48索引增量建议"], ["建议编号（接在 R48 之后，不重编既有编号）", "R48组", "条目", "建议处理", "R49表"]) + "\n",
          "## 5. 新表清单（表 G2）\n", tbl(S["G2_新表清单"], ["表", "任务", "来源", "版本", "数字核验", "行数"]) + "\n",
          "## 6. 包测试（表 A7；说明与耗时见 xlsx，报告只列检查名）\n", tbl(a7, ["检查", "通过"]) + "\n",
          "## 7. 数字溯源\n\n" + TRACE_MARK + "。\n"]
    sheets = {k: v for k, v in S.items() if isinstance(v, pd.DataFrame)}
    tc = trace_check(sheets)
    txt = "\n".join(L)
    assert txt.count(TRACE_MARK) == 1
    txt = txt.replace(TRACE_MARK, f"生成时逐值追踪 {tc['追踪数值']} 次格式化（{tc['不同数值']} 个不同原值），在报告引用表中找不到原值的 {len(tc['找不到'])} 个；"
                                  f"变异对照（{tc['变异对照数']} 个原值乘以 1 + 10⁻⁶）被放行 {tc['变异对照被放行']} 个；报告全文 SHA-256 登记在 `R49_NUM_TRACE.json`")
    OUT.mkdir(parents=True, exist_ok=True)
    toc = write_xlsx(S)
    MD.write_text(txt, encoding="utf-8")
    tc["报告sha"] = hashlib.sha256(MD.read_bytes()).hexdigest()[:16]
    (OUT / "R49_NUM_TRACE.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"md {len(txt)} 字；xlsx {len(toc)} 表；追踪 {tc['追踪数值']}，找不到 {len(tc['找不到'])}，变异放行 {tc['变异对照被放行']}")
    return not (tc["找不到"] or tc["变异对照被放行"])


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
