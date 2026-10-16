# -*- coding: utf-8 -*-
r"""R54：由 R54_QA.json、R54_PHASE1.json、R54_SUPP_INDEX.xlsx 生成给 GPT 的回复 R54_Claude_回复.md（数字全部从这些记录取）。"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

OUT = HERE.parent / "37_R54_补充材料v0.1"


def main():
    q = json.loads((OUT / "R54_QA.json").read_text(encoding="utf-8")); p1 = q["第一阶段"]
    idx = pd.read_excel(OUT / "R54_SUPP_INDEX.xlsx", sheet_name="1_Supplementary_index")
    smx = pd.read_excel(OUT / "R54_SUPP_INDEX.xlsx", sheet_name="2_Source_matrix_S0_R49_R51_R52")
    pend = pd.read_excel(OUT / "R54_SUPP_INDEX.xlsx", sheet_name="3_Pending_and_internal")
    dec = smx.v0_1_decision.str.split(":").str[0]
    used = int((dec == "USED").sum()); excl = int((dec == "EXCLUDED").sum()); idxonly = int(dec.str.startswith("NOT INCLUDED").sum())
    notpr = int((dec == "NOT PRINTED").sum())
    io = smx[smx.original_status == "仅索引"]; io_local = int(io.availability.str.startswith("original workbook present").sum())
    shortened = int(idx.fidelity.str.startswith("shortened").sum())
    a = []; A = a.append
    A("# R54 Claude 回复：v0.5a 只读核验 + Supplementary Information v0.1（来源约束汇编）\n")
    A("只读：没有拟合、没有重训、没有计算新的统计量；正文、六张主图与冻结结果一概未改。\n")
    A("## 0. 结论\n")
    A(f"1. **第一阶段全部通过。** v0.5 存档与 R53 交付的 v0.5 逐字节相同；v0.5a 在 {p1['比对元素']} 个段落/单元格中只改了 {p1['改动处']} 处，文字与 R53 复核给的替换句一致，全文数字集合不变。"
      f"新 S1 四个面板的纳排节点与 R49 C1、R48 S2/S9 逐个一致，结局框数字齐全；PNG 与 TIFF 像素逐点相同（TIFF {p1['S1_TIFF_dpi'][0].split('.')[0]} dpi、{p1['S1_TIFF_压缩']}）；图题与图注一致。")
    A(f"2. **Supplementary Information v0.1 已汇编**：Word 内含 {q['表数']} 张正式三线表（SM1、ST1–ST15、SM2 共 17 组）、补充图 S1 与两段补充方法；"
      f"另附机器可读长表 {q['长表数']} 张、共 {q['长表总行']:,} 行。每张表登记来源工作簿、工作表、版本、行数与是否删节（{shortened} 张为印刷版删节，全量在长表）。")
    A(f"3. **QA {q['QA']}**：印刷表中每个数字都在声明的来源列中找到；三张关键表与另一份冻结工作簿（R47 F2、B2）逐行交叉核对一致；无已被取代来源、无残留中文、无个人标识列；各项配变异对照。")
    A("4. **需要你裁决或知悉的 5 件事**见第 5 节；其中排版没有经过渲染目检（本机没有 Word/LibreOffice），请打开 Word 检查。\n")
    A("## 1. 第一阶段：v0.5a 与 S1\n")
    for c in p1["改动"]:
        o, n = " / ".join(c["old"]), " / ".join(c["new"])
        loc = n.split(":")[0]
        A(f"- {loc}：「…{o.split(': ', 1)[1][-110:]}」→「…{n.split(': ', 1)[1][-120:]}」")
    A(f"- 节点核对：{', '.join(f'{k} {chr(10003) if v else chr(10007)}' for k, v in p1['S1_节点与冻结来源一致'].items())}。S1 只改了布局与图题，没有改数字。\n")
    A("## 2. 来源矩阵（R48 S0 的 137 条 + R49/R51/R52 增量）\n")
    A(f"- 用上 {used} 条；已被取代 {excl} 条全部排除；仅索引 {idxonly} 条未纳入 v0.1（其中 {io_local} 条的原工作簿在本地可取，若你要可补）；可用但不印 {notpr} 条（理由逐条写在索引的 v0_1_decision 列，如与 ST8.4/ST9 重复、属于主图图源或本地路径）。")
    A("- S85（文献逐字摘录）只作内部依据，不进补充材料；S78（T2 逐重复合成记录）作为随代码发布的独立资源，不印成表。")
    A("- 所有数字只取自 R47/R48/R49/R51/R52 冻结工作簿与 R50 O5 计数；没有从旧 R34 流程表或 NWICU/NHANES v1 取任何人数或结果。\n")
    A("## 3. 补充材料构成\n")
    A("| 组 | 条目 | 印刷行数 | 保真度 |\n|---|---|---|---|")
    for r in idx.itertuples():
        A(f"| {r.group} | {r.item} {r.title.split('. ', 1)[1][:80]} | {r.printed_rows} | {'全量' if r.fidelity == 'full' else '删节（全量在长表）'} |")
    A("")
    A("## 4. 待补来源（PENDING_SOURCE）与内部项\n")
    for r in pend.itertuples():
        A(f"- {r.item}｜{r.status}｜{r.content}：{r.note}")
    A("")
    A("## 5. 需要你裁决或知悉\n")
    A("1. **编号**：v0.1 用 SM1、ST1.1 … 这类主题临时编号，刻意不用 S1–S95（那是内部索引号）。正式期刊编号请你在装订时统一。")
    A("2. **仅索引的原表**：例如 R46 D11（T2 预定对照 Q1–Q5）、R43 T1 各臂真值、R45 plasmode 原表，原工作簿都在本地；要不要补进 v0.2 由你定。")
    A("3. **T2 部署表**（ST12.3，104 行）我印进了 PDF，因为部署风险是核心结论；若期刊篇幅紧，可移到长表、正文只保留 Figure 6。")
    A("4. **两处写法约定**：T2 协变量下标按冻结代码从 0 开始（X0–X22，与冻结表 C1 的「Z·X₆」一致），并在补充方法中写明；23 项学习池取自运行代码实际读取的冻结特征池清单。")
    A("5. **排版未经渲染目检**：本机没有 Word/LibreOffice，只能按内容长度设置列宽并检查表格结构（表数、表题、首末格）。请你打开 Word 看横向页面、长表分页与字号。"
      "汇编中我自己抓到并修正了一处标签错误：ST4.2 树深的列名一度被读成位置序号（印成 Depth 1–4），已改为按列名取值并加断言。\n")
    A("## 6. QA 明细\n")
    A("| 检查 | 通过 | 变异对照 |\n|---|---|---|")
    for r in q["QA明细"]:
        A(f"| {r['检查']} | {'是' if r['通过'] else '否'} | {r['变异对照检出'] if isinstance(r['变异对照检出'], str) else ('检出' if r['变异对照检出'] else '未检出')} |")
    A("")
    A("## 7. ZIP 内容\n")
    A("- `PGSS_Supplementary_v0.1.docx`：补充材料 v0.1（横向 A4）。")
    A("- `PGSS_Supplementary_v0.1_DataTables.xlsx`：机器可读长表（README 页列出每张表对应的印刷条目与冻结来源）。")
    A("- `R54_SUPP_INDEX.xlsx`：补充条目索引、来源矩阵、待补与内部项、长表清单。")
    A("- `R54_QA.json`、`R54_PHASE1.json`：QA 与第一阶段核验记录。")
    A("- `code/`：汇编、构建、报告、打包脚本与 SHA256 清单。")
    A("\n这一批做完即停，等你逐组审核。")
    (OUT / "R54_Claude_回复.md").write_text("\n".join(a) + "\n", encoding="utf-8")
    print("written", len("\n".join(a)))


if __name__ == "__main__":
    main()
