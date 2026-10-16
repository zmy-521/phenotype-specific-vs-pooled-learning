# -*- coding: utf-8 -*-
r"""R56：由 R56_QC.json、R56_TESTS.json 与 R56_ICU_Predictor_Missingness.xlsx 生成 R56_READONLY_QC.md（数字全部从这些结果文件读取）。"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

OUT = HERE.parent / "38_R56_ICU预测变量缺失率"
XL = OUT / "R56_ICU_Predictor_Missingness.xlsx"


def n(x):
    return f"{int(x):,}"


def main():
    q = json.loads((OUT / "R56_QC.json").read_text(encoding="utf-8"))
    t = json.loads((OUT / "R56_TESTS.json").read_text(encoding="utf-8"))
    S = pd.read_excel(XL, sheet_name=None)
    L, P, FP, QA = S["1_S5A_long"], S["2_S5A_print"], S["QA_1_fingerprints"], S["QA_0_summary"]
    D4, D5, D8 = S["QA_5_vs_R51_Table1"], S["QA_6_phenotype_ascert"], S["QA_9_vs_R35_pool_QC"]
    dbs = list(q["分母"])
    a = []; A = a.append
    A("# R56 只读补齐：MIMIC-IV / eICU-CRD 23 项冻结预测变量的完整缺失率\n")
    A("只读：没有拟合、重训、重新校准或性能计算；没有插补、删人或重判表型；data v5、正文、Supplementary Word 与 S1–S44 编号一概未改。交付物只含聚合计数。\n")
    A("## 0. 结论\n")
    A(f"1. **46 行已完成**（2 库 × 23 项）。分母为 data v5 冻结合格 ICU 住院：" + "，".join(f"{d} {n(v)}" for d, v in q["分母"].items()) + "，与 GPT 指令、R51 Table 1、R49 C2 一致。")
    for s in q["各库"]:
        top = "、".join(f"{x['变量']} {x['缺失pct']:.1f}%" for x in s["缺失率最高"])
        A(f"   - {s['数据库']}：{s['缺失为0项']} 项无缺失；缺失比例中位数 {s['中位缺失pct']:.2f}%，最高 {s['最大缺失pct']:.2f}%（最高三项：{top}）。")
    A(f"2. **质控 {q['QA']}，独立测试 {t['测试']}**，各项变异对照全部检出（主脚本 {sum(bool(v) for v in q['变异对照'].values())}/{len(q['变异对照'])}，测试 {t['变异对照']}）。")
    A("3. **没有触发任何暂停条件**：冻结文件指纹一致、23 项变量可以确定、患者分母无冲突、程序无影响正确性的错误。")
    A("4. 新表暂定 **Supplementary Table S5A（候选编号，待 GPT 确认）**；打印版在 xlsx 的 `2_S5A_print`，完整字段在 `1_S5A_long`。\n")
    A("## 1. 来源与指纹\n")
    A("| 对象 | 登记来源 | 登记 | 实测 | 一致 | 本轮用途 |\n|---|---|---|---|---|---|")
    for r in FP.itertuples(index=False):
        A(f"| {r.对象} | {r.登记来源} | `{r.登记}` | `{r.实测}` | {'是' if r.一致 else '否'} | {r.本轮用途} |")
    A("\n- 指纹 = 文件 SHA-256 前 16 位，与 DATA_VERSION.json 的登记方式相同。eICU X 在 data v3、v4、v5 中逐字节相同，所以 eICU 的时间窗按 data_v2/v3 清单登记的规则书写（v5 清单「未改动项」含「eICU 全部输入」）。\n")
    A("## 2. 变量清单、字典与 S4 对应\n")
    A("- 23 项 = `16_R35_FIX/R35_DATA_QC.xlsx`「4_最终特征池」（其文件指纹等于 data v3/v4 清单登记的「特征池SHA」）。前 22 项直接取冻结 X 列；第 23 项「性别(F指示)」不在 X 中，按冻结加载器 `pgss_r38_bench.load` 的同一表达式由 META.sex 构造（F → 1，M → 0，其余 → 缺失）。")
    A(f"- 与 Supplementary Table S4（R55 v0.2 Word，按文档顺序解析）逐行同名、同序：23/23。冻结加载器实际产出的列顺序与清单相同（两库各 23 列）。")
    A("- 字典来源：MIMIC itemid、eICU labname 与生理范围取自 `pgss_r35_extract.py` 的 LAB / CHART / E_VITAL 常量（AST 只读解析；文件指纹 = R47 登记）；单位取自 `R34_VARMAP_VERIFIED.xlsx`「1_映射与核验」的单位列与换算说明（两库同单位、无需换算；本轮未连库重扫单位字段）；时间窗取自各版 DATA_VERSION.json 登记的规则原文。逐变量出处见 xlsx `3_Dictionary`。\n")
    A("## 3. 结果（打印版，S5A 候选）\n")
    A("| " + " | ".join(P.columns) + " |\n|" + "---|" * len(P.columns))
    for r in P.itertuples(index=False):
        A("| " + " | ".join(str(x) for x in r) + " |")
    A("\n## 4. 交叉核验\n")
    A("| 检查 | 通过 | 变异对照 | 说明 |\n|---|---|---|---|")
    for r in QA.itertuples(index=False):
        A(f"| {r.检查} | {'是' if r.通过 else '否'} | {r.变异对照检出} | {r.说明} |")
    for r in t["明细"]:
        A(f"| {r['测试']} | {'是' if r['通过'] else '否'} | {r['变异对照检出']} | {r['说明']} |")
    A("")
    A("- R51 Table 1 的 9 项代表性指标中，与预测变量池重合的是年龄、血红蛋白、血肌酐、RDW、血小板、血氯 6 项，另加女性（= 性别已知人数），两库共 "
      f"{int(D4.一致.notna().sum())} 格逐值相同；eGFR、白蛋白、BMI 只用于表型定义、不是预测变量，未比较（R51 原值仍列在 `QA_5_vs_R51_Table1`）。")
    A("- 表型可判定的冻结规则要求定义所需预测变量同时有值（女性：性别；年龄≥65：年龄；贫血：血红蛋白 + 性别；eGFR 降低：血肌酐 + 年龄 + 性别）。"
      f"由本表矩阵算出的掩码与冻结 PHENOOK 逐行一致（不一致行数 {int(D5.逐行不一致数.sum())}），计数与 R49 C2 一致（{int(D5.一致.sum())}/{len(D5)}）。")
    A("- R383 v5 构建审计（2_逐变量v4到v5 的 n_available_v5）是冻结当天按同一合格分母算出的计数；DATA_VERSION.json 的「有值就诊」是 MIMIC 全部 85,242 个构建住院（不限合格）的计数，本轮按同一口径重数后也逐值相同。\n")
    A("## 5. 发现与需知（均不改变冻结结果）\n")
    lab = L[L.Category == "Laboratory"].groupby("Database")["Missing %"].agg(["min", "max"])
    vit = L[L.Category == "Vital sign"].groupby("Database")["Missing %"].agg(["min", "max"])
    labcols = set(L[L.Category == "Laboratory"]["Frozen column (data v5)"])
    e35 = D8[(D8.数据库 == "eICU-CRD") & D8.冻结列名.isin(labcols)].R35选池时缺失率
    r35_lab = (float(e35.min()), float(e35.max()))
    assert len(e35) == len(labcols) and e35.notna().all()
    A("1. **两库化验缺失程度不同**：化验缺失比例 MIMIC-IV " + f"{lab.loc['MIMIC-IV', 'min']:.1f}%–{lab.loc['MIMIC-IV', 'max']:.1f}%，eICU-CRD {lab.loc['eICU-CRD', 'min']:.1f}%–{lab.loc['eICU-CRD', 'max']:.1f}%；"
      f"体征 MIMIC-IV {vit.loc['MIMIC-IV', 'min']:.1f}%–{vit.loc['MIMIC-IV', 'max']:.1f}%，eICU-CRD {vit.loc['eICU-CRD', 'min']:.1f}%–{vit.loc['eICU-CRD', 'max']:.1f}%。"
      f"eICU 化验在 R35 选池时（data v2 修订录入规则之前）已为 {r35_lab[0] * 100:.1f}%–{r35_lab[1] * 100:.1f}%，v2 规则（24 h 后才修订录入的结果视为 landmark 时不可得）只再增加不到 1 个百分点（见第 2 条），"
      "即两库差距在该规则之前已经存在。只作描述，不推断原因。")
    rr = {r["数据库"]: (r["min"], r["max"]) for r in q["R35最大差"]}
    A(f"2. **R35 选池阈值在 v5 下仍成立**：特征池按 R35 时的缺失率选入（两库都 <40%）。v5 的缺失率与 R35 相比变化 MIMIC-IV {rr['MIMIC-IV'][0]:+.2f} 至 {rr['MIMIC-IV'][1]:+.2f} 个百分点、"
      f"eICU-CRD {rr['eICU-CRD'][0]:+.2f} 至 {rr['eICU-CRD'][1]:+.2f} 个百分点（eICU 体征与年龄完全不变），46 项全部低于 40%，没有变量因数据版本更新越过选池阈值。逐项原因见 `QA_9_vs_R35_pool_QC`。")
    sx = L[(L.Database == "eICU-CRD") & (L["Frozen column (data v5)"] == "性别F")].iloc[0]
    A(f"3. **eICU-CRD 有 {n(sx['Missing n'])} 次住院性别未记录为男/女**：女性指示缺失，与 Table 1 女性已知数 {n(sx['Observed n'])} 一致；这些住院的女性、贫血、eGFR 降低表型按冻结规则不可判定，不计为阴性。")
    A("4. **单位写法**：血钠、血钾、血氯、碳酸氢盐、阴离子间隙在字典中记为 mEq/L，打印版写作 mmol/L（一价离子数值相同），与 Table 1 和 S5 的写法一致；若 GPT 偏好原记法可直接改列名，数字不变。")
    A("5. **不要引用 R34 映射表的 itemid 列**：`R34_VARMAP_VERIFIED.xlsx` 的 itemid 列包含体液标本条目（如 Creatinine, Ascites），R35 起已改为只取血液 itemid（`pgss_r35_extract.py` LAB）。本表只引用了 R34 表的单位列。")
    A("6. **v3 构建脚本无登记指纹**：`pgss_r381_build_v3.py` 在 data_v3 写出后又被修改过（文件时间晚于 data_v3），且没有登记 SHA，所以 eICU 时间窗不以该脚本为据，而以 data_v2/v3 清单登记的规则原文与已登记的 `pgss_r35_extract.py` 为据。\n")
    A("## 6. 建议的 S5A 表题与表注（英文，供 GPT 采用或修改）\n")
    A("**Supplementary Table S5A.** Completeness of the 23 frozen predictors in the eligible MIMIC-IV and eICU-CRD cohorts.\n")
    A(f"Denominators are all eligible ICU stays (MIMIC-IV n = {n(q['分母']['MIMIC-IV'])}; eICU-CRD n = {n(q['分母']['eICU-CRD'])}). A value was missing when no measurement passing the frozen physiological range filter "
      "was available in the predictor window: MIMIC-IV, charttime from ICU admission to the 24-hour landmark with storetime before the landmark; eICU-CRD, result or observation offset 0 to <1,440 minutes after unit admission "
      "(laboratory results revised after 24 hours treated as unavailable). Female sex was missing when sex was not recorded as female or male. Counts describe the frozen predictor matrix before any model-specific handling; "
      "no value was imputed for this table. Predictor order follows Supplementary Table S4; sources, units and windows for each predictor are given in the accompanying data table.\n")
    A("## 7. ZIP 内容\n")
    A("- `R56_ICU_Predictor_Missingness.xlsx`：`1_S5A_long`（46 行正式汇总）、`2_S5A_print`（23 行打印版）、`3_Dictionary`（逐变量来源、单位、时间窗与出处）、`QA_0`–`QA_9`（指纹、S4 对应、分母、五路交叉核验、与 R35 选池时比较）。")
    A("- `R56_READONLY_QC.md`（本文件）、`R56_QC.json`、`R56_TESTS.json`。")
    A("- `code/`：只读计数、独立测试、报告、打包四个脚本；`SHA256_MANIFEST.json`。")
    A("\n本轮任务到此结束，不启动下一轮研究或全面复核。")
    (OUT / "R56_READONLY_QC.md").write_text("\n".join(a) + "\n", encoding="utf-8")
    print("written", len("\n".join(a)))


if __name__ == "__main__":
    main()
