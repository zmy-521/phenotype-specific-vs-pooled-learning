# -*- coding: utf-8 -*-
r"""R43 图（静态 PNG，浅色；配色与 R42 相同：参考调色板槽位 1 蓝 #2a78d6、槽位 2 橙 #eb6834，身份另用形状区分）。
  F1 学习曲线：P* 与 HTFS **各自的绝对 Brier**（两轨等权），4 行（库 × 结局）× 6 列（表型）；
     横轴 = 目标训练比例（按簇抽取）；只画全部外层折可评价的点，缺点处断开。纵轴各面板独立（任务基线风险不同）。
  F2 学习曲线（R42 F2 重画，只改横轴名称为「目标训练比例，按簇抽取」并注明 eICU 实际比例高于目标）。
  F3 T1：各机制下 HTFS_D 相对 P*_D 的真值相对收益（逐重复点 + 均值），逐轨；1% 边界虚线。
  F4 T1：各选择器部署后（两轨等权）相对 P*_D 的真值改善，均值 ± 2 SE，按机制分面板、按 (n, π_A) 区分形状。
  F5、F6 同 F3、F4，但只画 T1 扩展（N1-T、P2-T，160 个重复，探索性），与原 320 分开。
用法：python pgss_r43_figures.py [1 2 3 4 5 6]
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
R43 = ROOT / "24_R43_结果补齐与T1"
OUT = R43 / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb", "#b5b4ae"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
OC = {"OUT_HOSP_MORT": "住院死亡", "OUT_LOS72": "ICU>72h"}
DBN = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD"}
PHENOS = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
FR = [0.25, 0.5, 0.75, 1.0]
XLAB = "目标训练比例（%，按簇抽取）"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig1():
    L = pd.read_excel(R43 / "readonly" / "R43_READONLY.xlsx", sheet_name="11_学习曲线绝对风险")
    L = L[L.完整度 >= 0.999]
    rows = [(db, oc) for db in ("mimic", "eicu") for oc in ("OUT_HOSP_MORT", "OUT_LOS72")]
    fig, axes = plt.subplots(4, 6, figsize=(22, 13.5), sharex=True)
    for i, (db, oc) in enumerate(rows):
        for j, ph in enumerate(PHENOS):
            ax = axes[i, j]; d = L[(L.库 == db) & (L.结局 == oc) & (L.表型 == ph)].set_index("目标比例").reindex(FR)
            for a, col, mk in (("P*", BLUE, "o"), ("HTFS", ORANGE, "s")):
                ax.plot([100 * f for f in FR], d[f"{a}_两轨等权"].to_numpy(float), color=col, marker=mk, ms=6, lw=2,
                        mec=SURF, mew=1.2)
            clean(ax); ax.set_xticks([25, 50, 75, 100])
            if i == 0:
                ax.set_title(ph, color=INK, fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{DBN[db]}\n{OC[oc]}\n绝对 Brier（两轨等权）", fontsize=11)
            if i == 3:
                ax.set_xlabel(XLAB, fontsize=10)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    h = [plt.Line2D([], [], color=BLUE, marker="o", lw=2, ms=6, label="P*（强合并参照）"),
         plt.Line2D([], [], color=ORANGE, marker="s", lw=2, ms=6, label="HTFS（完整专业化）")]
    fig.legend(handles=h, loc="upper right", frameon=False, ncol=2, fontsize=12)
    fig.suptitle("学习曲线：各模型自身的绝对 Brier（只画全部外层折可评价的点；eICU 25%/50% 无此类点）",
                 x=0.01, ha="left", color=INK, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "R43_F1_学习曲线绝对风险.png", dpi=130); plt.close(fig)


def fig2():
    x = pd.read_excel(ROOT / "23_R42_正式主实验" / "inference" / "R42_INFERENCE.xlsx", sheet_name=None)
    C = x["2_全部对照"]; LC = x["8_学习曲线"]
    full = C[(C.对照 == "E1") & (C.尺度 == "bal")][["库", "结局", "表型", "观察相对改善"]].rename(columns={"观察相对改善": "v"})
    full["比例"] = 1.0
    part = LC[["库", "结局", "表型", "比例", "HTFS_vs_P*_bal_相对", "完整度"]].rename(columns={"HTFS_vs_P*_bal_相对": "v"})
    part = part[part.完整度 >= 0.999].drop(columns="完整度")
    D = pd.concat([part, full])
    panels = [(db, oc) for db in ("mimic", "eicu") for oc in ("OUT_HOSP_MORT", "OUT_LOS72")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey=True)
    for ax, (db, oc) in zip(axes.ravel(), panels):
        d = D[(D.库 == db) & (D.结局 == oc)]
        W = d.pivot_table(index="表型", columns="比例", values="v").reindex(columns=FR)
        for ph, row in W.iterrows():
            ax.plot([100 * f for f in FR], 100 * row.to_numpy(float), color=GREY, lw=1.2, marker="o", ms=4)
        start, keep = None, None
        for i in range(len(FR) - 1, -1, -1):
            k = W[FR[i:]].dropna().index
            if len(k) >= 3:
                start, keep = i, k
        if start is not None:
            med = W.loc[keep, FR[start:]].median()
            ax.plot([100 * f for f in FR[start:]], 100 * med.to_numpy(float), color=BLUE, lw=2.5, marker="o", ms=8,
                    mec=SURF, mew=1.5)
            ax.text(0.98, 0.04, f"中位线：{len(keep)} 个任务，{int(100 * FR[start])}–100%", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color=INK2)
        ax.axhline(0, color=INK2, lw=1); ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_title(f"{DBN[db]} · {OC[oc]}", loc="left", color=INK)
        ax.set_xticks([25, 50, 75, 100]); clean(ax)
    for ax in axes[1]:
        ax.set_xlabel(XLAB)
    for ax in axes[:, 0]:
        ax.set_ylabel("相对 Brier 改善（%）\nHTFS 相对 P*（两轨等权）")
    h = [plt.Line2D([], [], color=GREY, lw=1.2, marker="o", ms=4, label="单个任务（只画全部外层折可评价的点）"),
         plt.Line2D([], [], color=BLUE, lw=2.5, marker="o", ms=8, label="中位数（同一组任务，见各面板右下注）"),
         plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label="预设实质边界 1%")]
    fig.legend(handles=h, loc="upper right", frameon=False)
    fig.suptitle("学习曲线：专业化收益随目标训练比例的变化（点估计，未画区间）", x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.005, "按不可拆分的簇抽取：eICU 的实际训练比例高于目标（实际人数、比例与簇数见证据表）。",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(OUT / "R43_F2_学习曲线相对改善.png", dpi=150); plt.close(fig)


SCN = {"N1": "N1 无关系差异", "N2": "N2 只有截距差异", "P1": "P1 交互（SZ 可学）", "P2": "P2 轨道特异稀疏结构",
       "N1-T": "N1-T（Z 由 X₈ 阈值定义）", "P2-T": "P2-T（Z 由 X₈ 阈值定义）"}
CONDS = [(6000, 0.15), (6000, 0.30), (12000, 0.15), (12000, 0.30)]
CMARK = {(6000, 0.15): ("o", BLUE), (6000, 0.30): ("s", BLUE), (12000, 0.15): ("o", ORANGE), (12000, 0.30): ("s", ORANGE)}


def fig3(runs="runs", out="R43_F3_T1真值收益.png", label="T1"):
    A = []
    for d in sorted((R43 / "t1" / runs).iterdir()):
        if (d / "_DONE").exists():
            a = pd.read_parquet(d / "arms.parquet").set_index("臂"); p = d.name.split("__")
            A.append(dict(机制=p[0], n=int(p[1][1:]), πA=int(p[2][2:]) / 100,
                          G_A=(a.loc["P*_D", "R_A"] - a.loc["HTFS_D", "R_A"]) / a.loc["P*_D", "R_A"],
                          G_B=(a.loc["P*_D", "R_B"] - a.loc["HTFS_D", "R_B"]) / a.loc["P*_D", "R_B"]))
    A = pd.DataFrame(A)
    scens = [s for s in SCN if s in set(A.机制)]
    fig, axes = plt.subplots(1, len(scens), figsize=(max(4.6 * len(scens), 13), 6.2), sharey=True, squeeze=False)
    for ax, sc in zip(axes[0], scens):
        for k, (n, pi) in enumerate(CONDS):
            for g, off in (("A", -0.17), ("B", 0.17)):
                v = 100 * A[(A.机制 == sc) & (A.n == n) & (A.πA == pi)][f"G_{g}"].to_numpy()
                if len(v) == 0:
                    continue
                xx = k + off + np.random.default_rng(k).uniform(-0.06, 0.06, len(v))
                col = BLUE if g == "A" else ORANGE; mk = "o" if g == "A" else "s"
                ax.plot(xx, v, ls="none", marker=mk, ms=4, color=col, alpha=0.45)
                ax.plot([k + off - 0.1, k + off + 0.1], [v.mean()] * 2, color=INK, lw=2)
        ax.axhline(0, color=INK2, lw=1); ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([f"n={n//1000}k\nπA={pi:.2f}" for n, pi in CONDS], fontsize=9)
        ax.set_title(SCN[sc], loc="left", color=INK, fontsize=12); clean(ax)
    axes[0, 0].set_ylabel("真值相对 Brier 收益（%）\nHTFS_D 相对 P*_D，正值 = 专业化更好")
    h = [plt.Line2D([], [], color=BLUE, marker="o", ls="none", ms=6, label="A 轨（逐重复）"),
         plt.Line2D([], [], color=ORANGE, marker="s", ls="none", ms=6, label="B 轨（逐重复）"),
         plt.Line2D([], [], color=INK, lw=2, label="均值"),
         plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label="边界 m = 1%")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.995, 0.93), frameon=False, ncol=4)
    fig.suptitle(f"{label}：已知机制下完整专业化相对强合并参照的真值收益（每条件 20 个独立训练重复）", x=0.01, ha="left",
                 color=INK, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(OUT / out, dpi=150); plt.close(fig)


SEL_ORDER = ["ALWAYS_POOLED", "ALWAYS_SPLIT", "CV_ANY|P*", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*",
             "CV_ANY|S0*", "CV_POINT_M|S0*", "CV_LB_M|S0*", "FCT|S0*", "MIX", "USTUN_ADAPT"]


def fig4(summary="R43_T1_SUMMARY.xlsx", out="R43_F4_T1选择器性能.png", label="T1"):
    T = pd.read_excel(R43 / "t1" / summary, sheet_name="D_选择器两轨等权")
    scens = [s for s in SCN if s in set(T.机制)]
    fig, axes = plt.subplots(1, len(scens), figsize=(max(4.8 * len(scens), 13), 7.8), sharey=True, squeeze=False)
    for ax, sc in zip(axes[0], scens):
        for k, (n, pi) in enumerate(CONDS):
            mk, col = CMARK[(n, pi)]
            for i, s in enumerate(SEL_ORDER):
                r = T[(T.机制 == sc) & (T.n == n) & (T.πA == pi) & (T.选择器 == s)]
                if r.empty:
                    continue
                r = r.iloc[0]; y = i + (k - 1.5) * 0.17
                ax.plot([100 * (r.均值 - 2 * r.SE), 100 * (r.均值 + 2 * r.SE)], [y, y], color=col, lw=1.5)
                ax.plot(100 * r.均值, y, marker=mk, ms=6, color=col, mec=SURF, mew=1, ls="none")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(SEL_ORDER))); ax.set_yticklabels(SEL_ORDER, fontsize=10)
        ax.set_ylim(len(SEL_ORDER) - 0.5, -0.6)
        ax.set_title(SCN[sc], loc="left", color=INK, fontsize=12)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_xlabel("相对 P*_D 的真值改善（%，两轨等权）")
    h = [plt.Line2D([], [], color=CMARK[c][1], marker=CMARK[c][0], ls="none", ms=6,
                    label=f"n={c[0]//1000}k, πA={c[1]:.2f}") for c in CONDS]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.995, 0.935), frameon=False, ncol=4)
    fig.suptitle(f"{label}：选择器部署后的实际性能（均值 ± 2 SE，20 个重复；0 = 与永远用 P*_D 相同）", x=0.01, ha="left",
                 color=INK, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT / out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    EXT = "T1 扩展（探索性）"
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4,
         "5": lambda: fig3("runs_ext", "R43_F5_T1扩展真值收益.png", EXT),
         "6": lambda: fig4("R43_T1_EXT_SUMMARY.xlsx", "R43_F6_T1扩展选择器性能.png", EXT)}[w]()
        print("F" + w, "已写")
