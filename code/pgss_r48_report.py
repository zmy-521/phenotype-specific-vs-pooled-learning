# -*- coding: utf-8 -*-
r"""R48 图稿与补充材料索引（GPT 第 52 轮）：xlsx 与 md **同源生成**。按第 52 轮要求不写新的科研裁决，只写交付清单、改动与核对。

输入 = pgss_r48_sources.build()（只读 R47 数字包、R42–R46 已交付结果表、T2 逐重复记录）；另读包测试结果 R48_PACKAGE_TESTS.json（若已存在）
与 GPT 复核包比对结果 R48_GPT_ORACLE_COMPARE.json（若已存在）。
数字溯源两层：① 舍入匹配（pgss_r43_numaudit，打包自检时执行）；② 生成时逐值追踪：pc/fx 格式化的每个原值必须在报告引用表中精确存在，
变异对照（原值 × (1 + 1e-6)）必须找不到；报告全文 SHA 登记在 R48_NUM_TRACE.json。
输出：30_R48_图稿与补充材料/R48_图稿与补充材料索引.md、R48_图源与补充材料.xlsx、R48_NUM_TRACE.json
用法：python pgss_r48_report.py
"""
import sys, json, pathlib, hashlib, re
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r48_sources as SRC

OUT = SRC.OUT
MD = OUT / "R48_图稿与补充材料索引.md"
XL = OUT / "R48_图源与补充材料.xlsx"
TRACE = []
TRACE_MARK = "〔逐值追踪结果〕"
FIGS = [("Fig1", "R48_Fig1_study_design", "研究设计（示意）", "F1_研究设计", "T2 框副标题收紧；选择程序一行注明在哪里运行；sample-proportion-weighted"),
        ("Fig2", "R48_Fig2_ICU_benchmark", "三个 ICU 库 36 个任务：HTFS 相对 P* 的两轨等权相对 Brier 改善", "F2_ICU复现", "横轴注明 pointwise；图内区间类型说明"),
        ("Fig3", "R48_Fig3_NHANES_attribution", "NHANES 低白蛋白：原比较与目标一致对照；绝对 Brier", "F3a_NHANES归因_相对；F3b_NHANES归因_绝对",
         "Population → Sample-proportion-weighted；图内区间类型说明"),
        ("Fig4", "R48_Fig4_learning_curves", "MIMIC-IV、eICU-CRD 学习曲线", "F4_学习曲线", "图内说明加「只有点估计」"),
        ("Fig5", "R48_Fig5_T2_mechanisms", "T2：HTFS_D 相对 P*_D 的逐轨真值相对收益", "F5_T2机制", "图内说明：模拟均值区间不是真实数据置信区间"),
        ("Fig6", "R48_Fig6_T2_selection", "T2：a 最终两轨等权改善；b 逐轨最终风险恶化；c 重复级至少一轨不达标启用", "F6a_T2部署两轨等权；F6b_T2逐轨实际变差；F6c_T2启用错误",
         "加入 CV_LB_M|P*；图内说明写明 FCT 部署 F 训练模型、b 不是 c 的错误启用率")]
CITED = ["A1_输入与交付副本比对", "A2_参照定义核对", "A3_取代声明核对", "A4_包测试", "A5_冻结脚本未改", "A6_GPT复核包比对", "D1_图索引",
         "G1_第52轮落实", "O1_有限D候选硬选择oracle", "S0_补充材料索引", "E1_环境依赖", "A7_区间常量"]
CAT = {"A": "核对", "D": "图索引", "E": "环境与路径", "F": "图源（图中每个点与每句图内说明）", "G": "第 52 轮落实", "O": "事后描述性补充", "S": "补充材料（索引与接回的表）"}


def pc(v, d=2):
    TRACE.append(float(v))
    return f"{100 * float(v):.{d}f}%"


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
                cells.append("—" if not np.isfinite(v) else str(int(v)) if float(v).is_integer() else f"{v:.3g}")
            else:
                cells.append(esc(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def a4_tests():
    p = OUT / "R48_PACKAGE_TESTS.json"
    if not p.exists():
        return pd.DataFrame([dict(检查="包测试尚未运行", 通过=np.nan, 说明="")])
    return pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))["检查"])


def a5_frozen():
    """冻结脚本未改：本轮代码闭包中凡在 R47_ENV.json 登记过的文件，当前 SHA 必须与登记值相同。"""
    env = json.loads((SRC.R47D / "R47_ENV.json").read_text(encoding="utf-8"))["code_sha"]
    code, _ = SRC.closure()
    rows = [dict(文件=f, R47登记SHA16=env[f], 当前SHA16=SRC.sha16((HERE / f).read_bytes())) for f in code if f in env]
    d = pd.DataFrame(rows); d["相同"] = d.R47登记SHA16 == d.当前SHA16
    new = [f for f in code if f not in env]
    return d, new


def a6_gpt():
    p = OUT / "R48_GPT_ORACLE_COMPARE.json"
    if not p.exists():
        return pd.DataFrame([dict(项目="GPT 复核包", 状态="未到本地（第 52 轮回复提到「随附复核包」，本地没有收到），未比对", 比较值=np.nan, 一致=np.nan, 最大绝对差=np.nan)])
    return pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))["rows"])


def d1_figs():
    rows = []
    for k, fn, desc, src, chg in FIGS:
        F = [s.strip() for s in src.split("；")]
        rows.append(dict(图=k, 文件=f"figures/{fn}.png；.tif；.pdf", 内容=desc, 源表=src, 源表行数=sum(len(S_[s]) for s in F), 相对R47的改动=chg))
    return pd.DataFrame(rows)


S_ = {}


def write_xlsx(S, path=XL):
    sh = {k: v for k, v in S.items() if isinstance(v, pd.DataFrame)}
    sh.update(S["_copies"])
    def key(k):
        m = re.match(r"([A-Z])(\d+)", k)
        return ("AGDFOSE".index(k[0]), int(m.group(2)) if m else 0, k)
    order = sorted(sh, key=key)
    toc = pd.DataFrame([dict(表=k, 类别=CAT[k[0]], 行数=len(sh[k]), 报告引用="是" if k in CITED else "否") for k in order])
    note = ["R48（GPT 第 52 轮）：只改展示与交付；不拟合、不模拟、不重新计算任何模型性能。",
            "F 开头 = 图源（pgss_r48_figures.py 只读这些表作图；F0 的「图内说明」原样印在图上）。F2–F5 的数值与 R47 数字包逐值相同；F6 新增 CV_LB_M|P* 的行取自 R47 C3、C4、C6b。",
            "S0 = 完整补充材料索引；「接回」的表以 S<编号>_<源表名> 复制在本工作簿，数值与源表逐值相同（包测试检查）。S 编号是建议编号，最终编号由 GPT 定。",
            "「已被取代」= 勿引用：旧版在后续修正后重算或被更完整的表逐位包含，取代者见同行。",
            "O1 = 有限 D 候选硬选择 oracle：事后描述性补充，不是 MIX、FCT 等全部方法的性能上界；未重跑 T2；不是预定的确认性结果。",
            "尺度标识符 pop（如 P*_pop）沿用冻结表；图与文字中的名称为 sample-proportion-weighted（按记录等权 = 样本中的轨比例，不是调查加权）。"]
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

    def found(x):
        return bool(np.any(np.abs(pool - x) <= 1e-12 * max(1.0, abs(x))))
    miss = sorted({x for x in TRACE if not found(x)})
    mut = [x * (1 + 1e-6) for x in sorted(set(TRACE)) if x != 0][:50]
    return dict(追踪数值=len(TRACE), 不同数值=len(set(TRACE)), 找不到=miss, 变异对照数=len(mut), 变异对照被放行=int(sum(found(x) for x in mut)))


def main():
    TRACE.clear()
    S = SRC.build()
    S_.clear(); S_.update(S)
    S["A4_包测试"] = a4_tests()
    S["A5_冻结脚本未改"], new_code = a5_frozen()
    S["A6_GPT复核包比对"] = a6_gpt()
    S["D1_图索引"] = d1_figs()
    a1, a2, a3, a4, a5, a6 = (S[k] for k in ("A1_输入与交付副本比对", "A2_参照定义核对", "A3_取代声明核对", "A4_包测试", "A5_冻结脚本未改", "A6_GPT复核包比对"))
    idx, o1, g1, e1, e2 = S["S0_补充材料索引"], S["O1_有限D候选硬选择oracle"], S["G1_第52轮落实"], S["E1_环境依赖"], S["E2_路径说明"]
    npass = int(sum(x is True or x == 1 for x in a4.通过)) if "通过" in a4 else 0
    L = ["# R48 图稿与补充材料索引\n",
         "> 由 `pgss_r48_report.py` 生成；md 与 `R48_图源与补充材料.xlsx` 同源。按第 52 轮要求只返改图稿、图源与完整补充材料索引，不是新一轮科研裁决报告。\n",
         "## 0. 先看这里\n",
         "**本轮性质**：只改展示与交付。没有新拟合、没有新模拟、没有重新计算任何模型性能；冻结目录与冻结脚本都没有改动。\n",
         f"**输入核对**：读取的 {len(a1)} 个输入，与交接目录或交付 zip 内的副本逐字节相同 {int(a1.逐字节相同.sum())}/{len(a1)}（表 A1）。"
         f"O1 用的参照与改善公式先在 T2 逐重复记录上重算了 R47 表 C3 的全部 {int(a2.行数.iloc[0])} 个两轨等权均值，与冻结表{'一致' if bool(a2.通过.iloc[0]) else '不一致'}（表 A2）。\n",
         f"**冻结脚本**：本轮代码闭包中在 R47 登记过的 {len(a5)} 个脚本，当前 SHA 与登记值相同 {int(a5.相同.sum())}/{len(a5)}（表 A5）；新增的只有本轮的 {len(new_code)} 个脚本："
         + "、".join(f"`{f}`" for f in new_code) + "。\n",
         f"**三件交付**：本报告；`R48_图源与补充材料.xlsx`（图源、图注要点、完整补充材料索引与接回的 S 表、O1、环境与路径，首页「目录」）；"
         "`R48_FIGURES_AND_CODE.zip`（六张图的 PNG、300 dpi TIFF、PDF，图源 CSV，代码，`requirements.txt`，`PATHS.md`，README）。\n",
         f"**包测试**：{npass}/{len(a4)} 项通过（表 A4，第 7 节）。\n",
         "## 1. 第 52 轮逐条落实\n", tbl(g1, ["条目", "GPT要求", "做法", "位置", "结果"]) + "\n",
         "## 2. 六张图\n",
         tbl(S["D1_图索引"], ["图", "文件", "内容", "源表", "源表行数", "相对R47的改动"]) + "\n",
         "图 2–5 没有整体重新设计，源数值没有改（F2–F5 与 R47 逐值相同，见包测试）。图 6 的 CV_LB_M|P* 用 Okabe–Ito 朱红、实心加号；"
         "与另外五个系列在红色盲、绿色盲模拟下都能区分，且每个系列另有不同形状的标记。\n",
         "## 3. 图注要点（表 F0）\n",
         "「图内说明」原样印在图上（图脚本从 F0 读取）；「图注必须包含的要点」供 GPT 写图注。\n"]
    for r in S["F0_图注要点"].itertuples():
        L.append(f"- **{r.图}**（{r.区间或计量类型}）" + (f"：图内 `{r.图内说明}`" if r.图内说明 else "") + f"。图注要点：{r.图注必须包含的要点}")
    L.append("")
    L += ["## 4. 有限 D 候选硬选择 oracle（表 O1；事后描述性补充）\n",
          f"**性质**：{SRC.ORACLE_NOTE}\n",
          "我从 R47 代码包内的 T2 逐重复记录（arms）只读派生，给出两个候选集：O-2 只在 P\\*_D 与 HTFS_D 之间取（与 R45 对 T1 定义的「两纯策略较优者」相同），"
          "O-7 在全部 7 个 D 臂之间取。用途是与 GPT 复核包逐值比对，确认我们说的是同一个量；正文用哪一个、怎么写，由 GPT 定。\n"]
    v = o1.copy()
    L.append(tbl(v, ["候选集", "条件", "两轨等权改善均值", "下", "上", "A轨改善均值", "B轨改善均值", "A轨取HTFS_D的重复", "B轨取HTFS_D的重复"],
                 fmt={c: pc for c in ("两轨等权改善均值", "下", "上", "A轨改善均值", "B轨改善均值")}) + "\n")
    L.append("**与 GPT 复核包的比对**（表 A6）：\n\n" + tbl(a6, list(a6.columns)) + "\n")
    cnt = idx.groupby(["组", "处理"]).size().unstack(fill_value=0)
    cols = [c for c in ("R47包内", "接回", "本轮新增", "仅索引", "已被取代", "缺口") if c in cnt.columns]
    cnt = cnt[cols].reset_index()
    cnt["_o"] = cnt.组.str.extract(r"^(\d+)").astype(int); cnt = cnt.sort_values("_o").drop(columns="_o")
    L += ["## 5. 完整补充材料索引（表 S0）\n",
          f"共 {len(idx)} 条。「接回」的 {int((idx.处理 == '接回').sum())} 张表复制进 xlsx（表名 S<编号>_<源表名>），与源表逐值相同；"
          f"「已被取代」{int((idx.处理 == '已被取代').sum())} 张写明取代者，其中「包含（逐位）」的声明已逐行核对（表 A3：" +
          "；".join(f"{r.旧表} → {r.取代表} 找不到的行 {r.找不到的行}" for r in a3.itertuples()) + "）。建议编号只是建议，最终编号由 GPT 定。\n",
          tbl(cnt, ["组"] + cols) + "\n",
          "**接回的表**（第 52 轮点名的完整对照、辅助性能、各模型自身校准都在内）：\n",
          tbl(idx[idx.处理 == "接回"], ["建议编号", "组", "来源", "工作表", "英文标题", "行数"]) + "\n",
          "**缺口**（没有算，只登记）：\n"]
    for r in idx[idx.处理 == "缺口"].itertuples():
        L.append(f"- {r.工作表}：{r.内容}。{r.备注}。")
    L += ["", "**已被取代（勿引用）**：\n", tbl(idx[idx.处理 == "已被取代"], ["来源", "工作表", "取代者", "取代方式"]) + "\n",
          "## 6. 代码、环境与路径\n",
          "冻结核心脚本保持原样（表 A5）。环境依赖（表 E1，从代码闭包的 import 实时扫描，版本取自运行环境）另写成 zip 内的 `requirements.txt`：\n",
          tbl(e1, ["包", "版本", "直接导入的文件数", "作用"]) + "\n",
          f"路径说明（表 E2，同样实时扫描，另写成 zip 内的 `PATHS.md`）：绝对路径 {int((e2.类型 == '绝对路径').sum())} 处"
          f"（其中文档字符串或注释中的示例 {int(e2.作用.str.startswith('文档字符串').sum())} 处，不执行），项目内目录引用 {int((e2.类型 == '项目内目录').sum())} 处，"
          f"环境变量 {int((e2.类型 == '环境变量').sum())} 组。真正需要复现者改的只有两类：NWICU 原始数据目录与 NHANES 辅助模块目录；"
          "项目内目录一律相对「代码目录的上一级」。\n",
          "## 7. 包测试（表 A4；每项的说明与耗时见 xlsx 表 A4，报告只列检查名，说明中的数字不进报告）\n", tbl(a4, ["检查", "通过"]) + "\n",
          "## 8. 请 GPT 知悉\n",
          "1. S0 有两个缺口只登记、没有计算：MIMIC-IV/eICU-CRD 的 data v5 终版纳排流程；NWICU、NHANES 辅助指标的跨 seed 汇总。二者都只需从冻结数据或已交付逐 seed 表只读计数或汇总，是否需要由你定。\n",
          "2. 冻结表与代码里的尺度标识符仍是 pop（如 P\\*_pop、HTFS_pop）；图和图注统一用 sample-proportion-weighted。正文是否把下标改名由你定，数值不受影响。\n",
          "3. 第 52 轮提到的 GPT 复核包没有随回复到达本地；O1 是我按同一名称独立复算的两种候选集，比对结果见第 4 节。\n",
          "## 9. 数字溯源\n\n" + TRACE_MARK + "。\n"]
    S["A4_包测试"] = a4
    sheets = {k: v for k, v in S.items() if isinstance(v, pd.DataFrame)}
    tc = trace_check(sheets)
    txt = "\n".join(L)
    assert txt.count(TRACE_MARK) == 1
    txt = txt.replace(TRACE_MARK, f"生成时逐值追踪 {tc['追踪数值']} 次格式化（{tc['不同数值']} 个不同原值），在报告引用表中找不到原值的 {len(tc['找不到'])} 个；"
                                  f"变异对照（{tc['变异对照数']} 个原值乘以 1 + 10⁻⁶）被放行 {tc['变异对照被放行']} 个；"
                                  "追踪记录同时登记报告全文的 SHA-256，打包自检据此检出生成后对报告的任何改动（`R48_NUM_TRACE.json`）")
    OUT.mkdir(parents=True, exist_ok=True)
    toc = write_xlsx(S)
    MD.write_text(txt, encoding="utf-8")
    tc["报告sha"] = hashlib.sha256(MD.read_bytes()).hexdigest()[:16]
    (OUT / "R48_NUM_TRACE.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "requirements.txt").write_text(SRC.requirements_txt(e1), encoding="utf-8")
    (OUT / "PATHS.md").write_text(SRC.paths_md(e2), encoding="utf-8")
    print(f"md {len(txt)} 字；xlsx {len(toc)} 表；追踪 {tc['追踪数值']}，找不到 {len(tc['找不到'])}，变异放行 {tc['变异对照被放行']}")
    if tc["找不到"] or tc["变异对照被放行"]:
        print("逐值追踪未通过"); return False
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
