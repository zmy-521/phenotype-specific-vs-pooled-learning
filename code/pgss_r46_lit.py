# -*- coding: utf-8 -*-
r"""R46 直接相关先例的原文定位表（GPT 第 50 轮 §8：只做原文定位，不做系统综述，不估计「弱参照研究占比」）。

每条摘录必须在本地原文全文（PDF 逐页抽取文本，或 PMC HTML 抽取文本）中**逐字找到**（只做空白、连字与断行连字符的规范化），
否则该行标「未在原文中找到」并整表不通过；页码由命中位置自动给出（HTML 来源给小节）。
主题：信息提供 / 架构比较 / 目标一致性 / 选择成本。「与本研究的关系」只作描述，不据 R45/R46 结果推断既往研究有缺陷。
原文 PDF 与抽取文本只留本地（28_R46_最终核验与T2/lit/，版权材料不打包）；表中登记每个来源文件的 SHA256 前 16 位。
用法：python pgss_r46_lit.py → 28_R46_最终核验与T2/lit/R46_LIT_TABLE.xlsx、.json
"""
import sys, json, pathlib, hashlib, re
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

LIT = HERE.parent / "28_R46_最终核验与T2" / "lit"
SRC = {
    "suriyakumar2023": ("Suriyakumar VM, Ghassemi M, Ustun B. When Personalization Harms Performance: Reconsidering the Use of Group Attributes in Prediction. ICML 2023", "arXiv:2206.02058v3", "suriyakumar2023.pdf"),
    "paes2022": ("Paes LM, Long CX, Ustun B, Calmon FP. On the Epistemic Limits of Personalized Prediction. NeurIPS 2022", "NeurIPS 2022 proceedings（Paper-Conference.pdf）", "paes2022.pdf"),
    "ustun2019": ("Ustun B, Liu Y, Parkes DC. Fairness without Harm: Decoupled Classifiers with Preference Guarantees. ICML 2019 (PMLR 97)", "PMLR 97:6373–6382", "ustun2019.pdf"),
    "wang2020": ("Wang H, Hsu H, Diaz M, Calmon FP. To Split or Not to Split: The Impact of Disparate Treatment in Classification", "arXiv:2002.04788", "wang2020_split.pdf"),
    "suresh2018": ("Suresh H, Gong JJ, Guttag JV. Learning Tasks for Multitask Learning: Heterogenous Patient Populations in the ICU. KDD 2018", "arXiv:1806.02878v1；doi:10.1145/3219819.3219930", "suresh2018.pdf"),
    "pfohl2022": ("Pfohl SR, Zhang H, Xu Y, Foryciarz A, Ghassemi M, Shah NH. A comparison of approaches to improve worst-case predictive model performance over patient subpopulations. Sci Rep 2022", "arXiv:2108.12250v2（预印本版本）", "pfohl2022.pdf"),
    "dwork2018": ("Dwork C, Immorlica N, Kalai AT, Leiserson M. Decoupled classifiers for fair and efficient machine learning. FAT* 2018", "arXiv:1707.06613", "dwork2018.pdf"),
    "guan": ("Guan Z, Parmigiani G, Patil P. Merging versus Ensembling in Multi-Study Prediction: Theoretical Insight from Random Effects", "arXiv:1905.07382", "guan_merging.pdf"),
    "mugambi2024": ("Best of Both Worlds: Bridging One Model for All and Group-Specific Model Approaches using Ensemble-based Subpopulation Modeling（PMC11141864）", "PMC11141864（HTML 全文）", "mugambi2024_PMC11141864.html"),
    "ltt2021": ("Angelopoulos AN, Bates S, Candès EJ, Jordan MI, Lei L. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", "arXiv:2110.01052", "ltt2021.pdf"),
    "sagawa2020": ("Sagawa S, Koh PW, Hashimoto TB, Liang P. Distributionally Robust Neural Networks for Group Shifts. ICLR 2020", "arXiv:1911.08731", "sagawa2020.pdf"),
}
ROWS = [
    ("suriyakumar2023", "信息提供", "by testing that every group expects more accurate predictions from a personalized compared to a generic model trained without their group attributes",
     "估计目标是「提供组别属性是否让每组获益」，参照为不含组别属性的 generic 模型；本研究的估计目标不同：在已提供表型信息的强合并参照之上，独立训练是否再增加价值。"),
    ("suriyakumar2023", "目标一致性", "Common criteria for model selection guide these decisions on the basis of population-level performance",
     "指出按总体表现做模型选择可使某组变差；本研究把同一问题放在比较程序上：合并参照的开发目标与评价尺度不一致时，比较会把开发目标的差异计入专业化。"),
    ("suriyakumar2023", "架构比较", "decoupling effectively personalizes the entire model development pipeline",
     "原文认为解耦模型的收益部分来自整个开发流程按组进行；与本研究 NHANES 目标一致性对照所检验的解释同向（专属模型的开发流程按本轨进行）。"),
    ("suriyakumar2023", "选择成本", "We split each dataset into a training sample (80%) to fit models, and a test sample (20%) to evaluate the gains of personalization",
     "用固定留出测试样本检验个性化收益（自助法检验、Bonferroni）；本研究的 FCT 同样留出认证集，并量化了留出带来的样本成本。"),
    ("paes2022", "选择成本", "we can only claim that personalization improves performance for each group who provides data when we explicitly limit the number of group attributes used by a personalized model",
     "给出逐组验证个性化收益的极小极大错误下界（0-1 损失、固定模型审计）；本研究在 Brier、算法级重训与每次只处理一个二分表型的设定下，报告认证的检出与样本成本。"),
    ("paes2022", "信息提供", "We introduce a metric called the benefit of personalization (BoP) to measure the smallest gain in accuracy that any group expects to receive from a personalized model",
     "BoP 是相对 generic 模型的最小组收益；本研究逐轨报告相对强合并参照的收益，两者参照不同。"),
    ("ustun2019", "架构比较", "A pooled classifier that encodes group membership may not perform as well as a pair of decoupled classifier when we fit classifiers from a hypothesis class that cannot represent the heterogeneity between groups",
     "原文讨论了含组别编码的合并分类器与解耦分类器的比较，差异来自假设类能否表示组间异质性；本研究的合并族包含按表型分裂的树模型（SZ、SZ-GFS、SZ-union）。"),
    ("ustun2019", "目标一致性", "outputs the best classifier for A which is the worst classifier for",
     "原文图 1：合并训练按经验风险最小化服务多数组（tyranny of the majority）；与本研究「合并参照的开发目标决定其在小轨上的表现」同类。"),
    ("ustun2019", "选择成本", "using a hypothesis test on a hold-out dataset. We use an exact version of the McNemar test",
     "解耦决策用留出数据上的检验（rationality、envy-freeness）；本研究的 USTUN_ADAPT 只是适配（保留 rationality 一项、Brier 损失），不称复现。"),
    ("wang2020", "架构比较", "we compare the performance of split classifiers (i.e., classifiers trained and deployed separately on each group) with group-blind classifiers (i.e., classifiers which do not use a sensitive attribute)",
     "参照为 group-blind 分类器；本研究的参照含表型信息。"),
    ("wang2020", "选择成本", "In the finite sample regime, splitting is not necessarily beneficial and we provide data-dependent bounds to understand this effect",
     "有限样本下拆分不一定有益；本研究的学习曲线与 T1/T2 在给定机制下给出数值结果。"),
    ("suresh2018", "架构比较", "We compared our multi-task learning approach against two single-task approaches: 1) a separate single task model for each group, and 2) a global model for all patients, agnostic to task membership",
     "ICU 死亡预测中比较多任务、按组单独、全局（不含组别）模型；本研究的合并参照含表型信息，并比较独立训练相对它的增量。"),
    ("suresh2018", "目标一致性", "overall micro-measures can hide underperformance on specific subgroups",
     "强调逐组评价；本研究逐轨、两轨等权与人口加权三种尺度并报。"),
    ("pfohl2022", "架构比较", "We compare empirical risk minimization (ERM; the standard learning paradigm) applied to the entire training dataset with four alternatives: (1) training a separate model for each subpopulation",
     "EHR 预测中把合并 ERM 与按亚组单独训练等方案比较；与本研究的架构比较直接相关。"),
    ("pfohl2022", "目标一致性", "We evaluate the worst-case early-stopping rules in conjunction with worst-case model selection criteria that select hyperparameters based on the best worst-case performance on a held-out validation set",
     "原文已把合并模型的早停与模型选择改为按亚组最坏情况目标；本研究的 P*_pop / P*_B 同样改变合并参照的早停与选择目标（目标为人口加权或 B 轨），用于专业化收益的归因，而不是最坏组优化。"),
    ("pfohl2022", "架构比较", "no approach performs better, for each patient subpopulation examined, than standard learning procedures using the entire training dataset",
     "原文结论：多数情况下没有方案在每个亚组上优于使用全部训练数据的标准程序；与本研究 ICU 三库结果方向一致（估计目标、模型与指标不同）。"),
    ("dwork2018", "架构比较", "We provide a simple and efficient decoupling technique, that can be added on top of any black-box machine learning algorithm, to learn different classifiers for different groups",
     "解耦分类器的一般框架（联合损失、可含公平约束）；本研究的专属模型 HTFS 是逐轨独立开发的一种实现。"),
    ("dwork2018", "信息提供", "Transfer learning is used to mitigate the problem of having too little data on any one group",
     "以迁移学习缓解小组数据不足；本研究的 MIX 与合并族是在小轨上借用两轨信息的两种方式。"),
    ("guan", "架构比较", "there exists a transition point beyond which ensembling outperforms merging",
     "多研究预测中合并与分研究集成的转折点（岭回归理论）；本研究在表型轴、二分类结局与 XGBoost 下给出经验比较。"),
    ("mugambi2024", "架构比较", "It consistently outperforms one model for all and one model for each subgroup approaches",
     "集成式亚群模型与「一个模型」「每组一个模型」比较；原文的「一个模型」包含性别、种族等变量（R39–R40 已核对）。"),
    ("mugambi2024", "选择成本", "the personalization benefits gained from these models tradeoff their statistical power, and can be impractical when the subpopulation’s sample size is small",
     "亚群模型的样本量代价；本研究的学习曲线与认证样本成本给出同类代价的数值。"),
    ("ltt2021", "选择成本", "Our main insight is to reframe the risk-control problem as multiple hypothesis testing",
     "留出校准数据上的多重检验以控制风险；本研究 FCT 的「拟合—认证—测试」与之同属留出验证思路，但认证的是逐轨相对收益。"),
    ("sagawa2020", "目标一致性", "allows us to learn models that instead minimize the worst-case training loss over a set of pre-defined groups",
     "按组最坏情况改变训练目标；本研究不做最坏组优化，只改变合并参照的开发目标以匹配评价尺度。"),
    ("sagawa2020", "目标一致性", "or early stopping—we achieve substantially higher worst-group accuracies",
     "原文指出正则化与早停对组层面泛化的作用；与本研究「早停权重属于开发目标的一部分」相关。"),
]


def norm(s):
    s = s.replace("ﬁ", "fi").replace("ﬀ", "ff").replace("ﬂ", "fl").replace("ﬃ", "ffi").replace("ﬄ", "ffl")
    s = re.sub(r"-\s*\n\s*", "", s)                 # 断行连字符
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s)


def load(key):
    f = SRC[key][2]
    stem = pathlib.Path(f).stem
    if f.endswith(".html"):
        t = (LIT / "txt" / "mugambi2024.txt").read_text(encoding="utf-8")
        parts = re.split(r"=====SECTION ", t)
        segs = [(p.split("\n", 1)[0].strip()[:40] or "正文", p) for p in parts]
    else:
        t = (LIT / "txt" / (stem + ".txt")).read_text(encoding="utf-8")
        parts = re.split(r"=====PAGE (\d+)=====", t)
        segs = [(f"第 {parts[i]} 页", parts[i + 1]) for i in range(1, len(parts), 2)]
    return [(loc, norm(s)) for loc, s in segs], stem


def main():
    rows = []
    for key, theme, quote, rel in ROWS:
        segs, stem = load(key)
        q = norm(quote)
        hit = [loc for loc, s in segs if q in s]
        if not hit:                                   # 跨页：在相邻两段拼接中找
            hit = [f"{segs[i][0]}–{segs[i+1][0]}" for i in range(len(segs) - 1) if q in segs[i][1] + " " + segs[i + 1][1]]
        src = LIT / "pdf" / SRC[key][2]
        rows.append(dict(编号=key, 文献=SRC[key][0], 标识=SRC[key][1], 主题=theme, 原文位置="；".join(hit) if hit else "未在原文中找到",
                         原文摘录=quote, 与本研究的关系=rel, 核对级别="本地原文全文逐字匹配" if hit else "未通过",
                         来源文件SHA16=hashlib.sha256(src.read_bytes()).hexdigest()[:16]))
    T = pd.DataFrame(rows)
    # 变异对照：改一个词的摘录必须找不到
    segs, _ = load("pfohl2022")
    mut = not any(norm("no approach performs worse, for each patient subpopulation examined") in s for _, s in segs)
    ok = bool((T.核对级别 == "本地原文全文逐字匹配").all() and mut)
    summ = dict(全部逐字匹配=ok, 条目=len(T), 文献=int(T.编号.nunique()), 主题计数=T.主题.value_counts().to_dict(), 变异对照_改词摘录找不到=mut,
                范围说明="直接相关先例的原文定位，不是系统综述；不估计弱参照研究占比；关系列只作描述",
                未纳入="Vergouwe 2017（Stat Med，闭合检验选更新方法）：本轮未取得全文，不做原文定位")
    with pd.ExcelWriter(LIT / "R46_LIT_TABLE.xlsx", engine="openpyxl") as xw:
        T.to_excel(xw, sheet_name="先例原文定位", index=False)
    (LIT / "R46_LIT_TABLE.json").write_text(json.dumps(dict(summ, 表=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1)); print(T[["编号", "主题", "原文位置", "核对级别"]].to_string())
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
