# -*- coding: utf-8 -*-
r"""R47 六张主图（精简版，英文标注，供投稿稿件使用）。**只读 R47_冻结数字包.xlsx 的 F 开头图源表**，图中每个点都来自这些表。

  Fig1 研究设计示意（数字来自 F1_研究设计）
  Fig2 三个 ICU 库 36 个任务：HTFS 相对 P* 的两轨等权相对 Brier 改善与 95% 区间（F2_ICU复现）
  Fig3 NHANES 低白蛋白：原比较与目标一致对照（F3a 相对、F3b 绝对）
  Fig4 学习曲线：MIMIC-IV、eICU-CRD（F4_学习曲线；只画全部外层折可评价的点）
  Fig5 T2 已知机制：HTFS_D 相对 P*_D 的逐轨真值收益（F5_T2机制）
  Fig6 T2 选择与部署：两轨等权部署改善、逐轨实际变差、重复级不达标启用（F6a–F6c）
输出：29_R47_写作交接/figures/R47_FigN_*.png（300 dpi）与 .pdf。
用法：python pgss_r47_figures.py [xlsx 路径] [输出目录]
"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = HERE.parent
XL = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx"
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "29_R47_写作交接" / "figures"
INK, INK2, GRID, BG = "#1a1a1a", "#555555", "#e2e2e2", "#ffffff"
OI = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#999999"]      # Okabe–Ito
TA, TB = "#D55E00", "#0072B2"                                                          # 轨 A、轨 B
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8, "figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.fontsize": 8, "pdf.fonttype": 42})
COND = ["N1", "N1-T", "N1-H", "N2", "P1", "P2", "P2-T", "P2-H"]
COND_LAB = {"N1": "N1\nno track\ndifference", "N1-T": "N1-T\nthreshold Z", "N1-H": "N1-H\nheavy tails", "N2": "N2\nintercept\nonly",
            "P1": "P1\nphenotype\ninteraction", "P2": "P2\ntrack-specific\nstructure", "P2-T": "P2-T\nthreshold Z", "P2-H": "P2-H\nheavy tails"}
SEL = [("ALWAYS_SPLIT", "Always split (HTFS_D)", OI[0], "o"), ("CV_POINT_M|P*", "CV point estimate ≥ m", OI[3], "D"),
       ("FCT|P*", "Fit–certify–test (FCT)", OI[4], "s"), ("MIX", "MIX (pooled + track-specific blend)", OI[2], "^"),
       ("POOLMIX", "POOLMIX (pooled-only blend)", OI[6], "v")]
M = 1.0
sheets = {}


def rd(name):
    if name not in sheets:
        sheets[name] = pd.read_excel(XL, sheet_name=name[:31])
    return sheets[name]


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)


def refs(ax, axis="x", m=True):
    f = ax.axvline if axis == "x" else ax.axhline
    f(0, color=INK2, lw=0.9, zorder=1)
    if m:
        f(M, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ── Fig 1 ──────────────────────────────────────────────────────
def fig1():
    F = rd("F1_研究设计")
    v = lambda lab: int(F[F.标签 == lab].数值.iloc[0])
    fig = plt.figure(figsize=(7.2, 3.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50.5); ax.axis("off")

    def box(x, y, w, h, title, sub, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.0))
        ax.text(x + 1.0, y + h - 1.2, title, fontsize=8.2, fontweight="bold", va="top", color=INK)
        ax.text(x + 1.0, y + h - 4.6, sub, fontsize=7.2, style="italic", va="top", color=INK2)
        ax.text(x + 1.0, y + h - 8.4, body, fontsize=7.4, va="top", color=INK, linespacing=1.4)
    NL = chr(10)
    ax.text(1, 49.8, NL.join(["Does phenotype-specific training (HTFS) add predictive value",
                              "beyond a phenotype-aware pooled reference (P*)?"]),
            fontsize=9, fontweight="bold", va="top", color=INK, linespacing=1.3)
    nO, nP = v("outcomes per database"), v("phenotypes per database")
    box(1, 23, 31, 20, "L1  ICU benchmark", "pre-specified; same frozen program",
        NL.join([f"MIMIC-IV: {v('MIMIC-IV')} tasks", f"eICU-CRD: {v('eICU-CRD')} tasks", f"NWICU: {v('NWICU')} tasks",
                 f"{nO} outcomes × {nP} phenotypes each"]), "#eef4fb", TB)
    box(34.5, 23, 31, 20, "L2  Cross-domain replication", "same program, general population",
        NL.join([f"NHANES: {v('NHANES')} tasks", "10-year all-cause and", "cardiovascular death"]), "#f3f3f3", INK2)
    box(68, 23, 31, 20, "L3  Attribution control", "registered after L2 results; one task",
        NL.join([f"NHANES low albumin ({v('NHANES low albumin')} task)", "pooled program re-developed for", "the evaluation objective",
                 "(track B; population-weighted)"]), "#fbf1e9", TA)
    box(1, 1.2, 48, 17.5, "Compared programs", "frozen XGBoost engine",
        NL.join(["P*: best pooled candidate (with/without explicit", "      phenotype, calibration, track weights)",
                 "HTFS: each track developed separately", "Selection: always pooled/split, CV rules, FCT, MIX, POOLMIX"]), "#ffffff", INK2)
    box(51.5, 1.2, 47.5, 17.5, "T2  Independent simulation", "known truth; selection procedures applied as in real data",
        NL.join([f"{v('conditions')} conditions × {v('replicates per condition')} replicates; {v('selection procedures')} procedures",
                 "truth = conditional Brier risk on independent", "evaluation draws (≤ 10⁶ per replicate)"]), "#eef7f1", OI[2])
    ax.text(1, 20.6, "Metric: Brier risk by track (A = phenotype-positive, B = rest), balanced (A + B)/2 and population-weighted; margin m = 1% (relative)",
            fontsize=7.2, color=INK2, va="center")
    save(fig, "R47_Fig1_study_design")


# ── Fig 2 ──────────────────────────────────────────────────────
def fig2():
    F = rd("F2_ICU复现")
    dbs = ["MIMIC-IV", "eICU-CRD", "NWICU"]
    ocs = ["In-hospital death", "ICU stay >72 h"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.6), sharey=True)
    lo = 100 * F.相对_P2_5.min(); hi = max(100 * F.相对_P97_5.max(), M)
    ylab, ypos = [], []
    y = 0
    layout = []
    for oc in ocs:
        layout.append(("header", oc, y)); y += 1
        for ph in ["Female", "Age ≥65", "Anemia", "Reduced eGFR", "Low albumin", "Obesity"]:
            layout.append(("row", (oc, ph), y)); y += 1
        y += 0.4
    for ax, db in zip(axes, dbs):
        d = F[F.Database == db]
        for kind, key, yy in layout:
            if kind == "row":
                r = d[(d.Outcome == key[0]) & (d.Phenotype == key[1])]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([100 * r.相对_P2_5, 100 * r.相对_P97_5], [yy, yy], color=TB, lw=1.4, solid_capstyle="butt")
                ax.plot(100 * r.观察相对改善, yy, "o", color=TB, ms=4.2, mec="white", mew=0.6, zorder=3)
        refs(ax); clean(ax)
        ax.set_title(db); ax.set_xlim(lo - 0.3, hi + 0.4)
        ax.text(M + 0.07, 13.9, "m = 1%", fontsize=7.2, color=INK2, ha="left", va="bottom", rotation=90)
    for kind, key, yy in layout:
        if kind == "row":
            ylab.append(key[1]); ypos.append(yy)
        else:
            axes[0].text(-0.02, yy, key, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=9,
                         fontweight="bold", color=INK)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(ylab)
    axes[0].set_ylim(y - 0.2, -0.8)
    fig.supxlabel("Relative Brier improvement of HTFS over P*, % (balanced over tracks; 95% interval)     "
                  "← HTFS worse  |  HTFS better →", fontsize=8.5, y=0.01)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.03, 1, 1))
    save(fig, "R47_Fig2_ICU_benchmark")


# ── Fig 3 ──────────────────────────────────────────────────────
def fig3():
    A = rd("F3a_NHANES归因_相对"); B = rd("F3b_NHANES归因_绝对")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rows = [("B", "E1", "Track B, original", TA), ("B", "D1", "Track B, aligned", TB),
            ("pop", "E1", "Population, original", TA), ("pop", "C1", "Population, aligned", TB)]
    ys = [0, 1, 2.6, 3.6]
    for (sc, c, lab, col), yy in zip(rows, ys):
        r = A[(A.尺度 == sc) & (A.对照 == c)].iloc[0]
        a1.plot([100 * r.相对_P2_5, 100 * r.相对_P97_5], [yy, yy], color=col, lw=1.6)
        a1.plot(100 * r.观察相对改善, yy, "o", color=col, ms=5, mec="white", mew=0.6, zorder=3)
        a1.text(100 * r.观察相对改善, yy - 0.22, f"{100 * r.观察相对改善:.2f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
    a1.set_yticks(ys); a1.set_yticklabels([r[2] for r in rows], fontsize=8); a1.invert_yaxis()
    refs(a1); clean(a1); a1.set_xlim(-1.0, 4.3); a1.set_ylim(4.1, -0.7)
    a1.text(M, -0.65, "m = 1%", fontsize=7.2, color=INK2, ha="left", va="top")
    a1.set_xlabel("Relative Brier improvement of HTFS, %\n(95% interval)")
    a1.set_title("a  HTFS advantage", loc="left")
    fig.text(0.01, 0.005, "Original: P* developed for the balanced objective (frozen engine).  Aligned: pooled program developed for the "
             "evaluated scale (track B: HTFS_B vs P*_B; population: HTFS_pop vs P*_pop).", fontsize=6.8, color=INK2, ha="left", va="bottom")
    lab = {"R_P*（两轨等权开发）": "P* (balanced objective)", "R_P*_目标": "P* (aligned objective)", "R_HTFS_目标": "HTFS (aligned calibration choice)",
           "R_HTFS（原）": "HTFS (original)"}
    cols = {"R_P*（两轨等权开发）": TA, "R_P*_目标": TB, "R_HTFS_目标": OI[2], "R_HTFS（原）": OI[6]}
    for i, (t, ttl) in enumerate((("B", "Track B"), ("pop", "Population"))):
        d = B[B.目标 == t]
        for j, (mdl, nm) in enumerate(lab.items()):
            v = d[d.模型 == mdl].Brier.iloc[0]
            a2.plot(i + (j - 1.5) * 0.17, v, "o", color=cols[mdl], ms=6, mec="white", mew=0.6, label=nm if i == 0 else None)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["Track B", "Population"]); a2.set_xlim(-0.6, 1.6)
    a2.set_ylabel("Brier risk (absolute)"); clean(a2, grid="y")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=6.8, handletextpad=0.2, columnspacing=0.8)
    a2.set_title("b  Absolute Brier risk", loc="left")
    fig.tight_layout(w_pad=1.5, rect=(0, 0.05, 1, 1))
    save(fig, "R47_Fig3_NHANES_attribution")


# ── Fig 4 ──────────────────────────────────────────────────────
def fig4():
    F = rd("F4_学习曲线"); F = F[F.可画]
    phs = ["Female", "Age ≥65", "Anemia", "Reduced eGFR", "Low albumin", "Obesity"]
    pan = [("MIMIC-IV", "In-hospital death"), ("MIMIC-IV", "ICU stay >72 h"), ("eICU-CRD", "In-hospital death"), ("eICU-CRD", "ICU stay >72 h")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    for ax, (db, oc) in zip(axes.ravel(), pan):
        d = F[(F.Database == db) & (F.Outcome == oc)]
        for k, ph in enumerate(phs):
            e = d[d.Phenotype == ph].sort_values("比例")
            ax.plot(100 * e.比例, 100 * e["HTFS_vs_P*_bal_相对"], "-o", color=OI[k], ms=3.6, lw=1.3, label=ph)
        refs(ax, axis="y"); clean(ax, grid="y")
        ax.set_title(f"{db} · {oc}", loc="left", fontsize=9)
        ax.set_xticks([25, 50, 75, 100])
    axes[0, 1].text(100, M + 0.06, "m = 1%", fontsize=7.2, color=INK2, va="bottom", ha="right")
    fig.supylabel("HTFS vs P*, relative Brier improvement, % (balanced over tracks)", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=6, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.53, 1.0), handlelength=1.6, columnspacing=1.0)
    fig.text(0.07, 0.005, "Points are shown only where every outer fold was evaluable at that fraction "
             "(no such points for eICU-CRD at 25% and 50%); NWICU and NHANES had no learning-curve runs.",
             fontsize=6.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0.01, 0.07, 1, 0.95))
    p0, p1 = axes[1, 0].get_position(), axes[1, 1].get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.075, "Target training fraction, % (cluster-sampled)", fontsize=9, ha="center", va="top")
    save(fig, "R47_Fig4_learning_curves")


# ── Fig 5 ──────────────────────────────────────────────────────
def fig5():
    F = rd("F5_T2机制")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for g, col, off in (("A", TA, -0.16), ("B", TB, 0.16)):
        for i, c in enumerate(COND):
            r = F[(F.条件 == c) & (F.轨 == g)].iloc[0]
            x = i + off
            ax.plot([x, x], [100 * r.P5, 100 * r.P95], color=col, lw=0.8, alpha=0.55)
            ax.plot([x, x], [100 * r.下, 100 * r.上], color=col, lw=3.2, solid_capstyle="butt")
            ax.plot(x, 100 * r.真值相对收益均值, "o", color=col, ms=4.5, mec="white", mew=0.6, zorder=3,
                    label=(f"Track {g}" + (" (phenotype-positive, 15%)" if g == "A" else "")) if i == 0 else None)
    refs(ax, axis="y"); clean(ax, grid="y")
    ax.set_xticks(range(len(COND))); ax.set_xticklabels([COND_LAB[c] for c in COND], fontsize=7.4)
    ax.set_ylabel("True relative Brier gain of HTFS_D\nover P*_D, %")
    ax.text(-0.45, M + 0.1, "m = 1%", fontsize=7.5, color=INK2, va="bottom", ha="left")
    ax.legend(frameon=False, loc="upper left", fontsize=7.6)
    ax.text(0.0, -0.36, "Point: mean of 200 replicates; thick bar: mean ± 1.96 SE; thin bar: 5th–95th percentile across replicates.",
            transform=ax.transAxes, fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "R47_Fig5_T2_mechanisms")


# ── Fig 6 ──────────────────────────────────────────────────────
def fig6():
    A = rd("F6a_T2部署两轨等权"); B = rd("F6b_T2逐轨实际变差"); C = rd("F6c_T2启用错误")
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 0.85], hspace=0.62, wspace=0.18, top=0.9)
    ax = fig.add_subplot(gs[0, :])
    w = 0.15
    for j, (s, nm, col, mk) in enumerate(SEL):
        for i, c in enumerate(COND):
            r = A[(A.条件 == c) & (A.选择器 == s)].iloc[0]; x = i + (j - 2) * w
            ax.plot([x, x], [100 * r.下, 100 * r.上], color=col, lw=1.6)
            ax.plot(x, 100 * r.两轨等权部署改善均值, mk, color=col, ms=4.4, mec="white", mew=0.5, zorder=3, label=nm if i == 0 else None)
    refs(ax, axis="y", m=False); clean(ax, grid="y")
    ax.set_xticks(range(len(COND))); ax.set_xticklabels(COND, fontsize=8)
    ax.set_ylabel("Balanced improvement\nover P*_D, % (mean ± 1.96 SE)")
    ax.set_title("a  Final risk after selection (balanced over tracks)", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    for k, (g, ttl) in enumerate((("A", "b  Replicates with worse track-A risk"), ("B", "   Replicates with worse track-B risk"))):
        bx = fig.add_subplot(gs[1, k])
        for j, (s, nm, col, mk) in enumerate(SEL):
            for i, c in enumerate(COND):
                r = B[(B.条件 == c) & (B.选择器 == s) & (B.轨 == g)].iloc[0]; x = i + (j - 2) * w
                p = r.部署实际变差_分子 / r.部署实际变差_分母
                bx.plot([x, x], [100 * r.部署实际变差_CP下, 100 * r.部署实际变差_CP上], color=col, lw=1.0)
                bx.plot(x, 100 * p, mk, color=col, ms=3.4, mec="white", mew=0.4, zorder=3)
        clean(bx, grid="y"); bx.set_ylim(-3, 103)
        bx.set_xticks(range(len(COND))); bx.set_xticklabels(COND, fontsize=7.2, rotation=0)
        bx.set_title(ttl, loc="left"); bx.set_ylabel("% of 200 (CP 95% CI)" if k == 0 else "")
    cx = fig.add_subplot(gs[2, :])
    for j, (s, nm, col, mk) in enumerate([x for x in SEL if x[0] in ("CV_POINT_M|P*", "FCT|P*")]):
        for i, c in enumerate(COND):
            r = C[(C.条件 == c) & (C.选择器 == s)].iloc[0]; x = i + (j - 0.5) * 0.22
            p = r["≥1轨不达标启用_分子"] / r["≥1轨不达标启用_分母"]
            cx.plot([x, x], [100 * r["≥1轨不达标启用_CP下"], 100 * r["≥1轨不达标启用_CP上"]], color=col, lw=1.2)
            cx.plot(x, 100 * p, mk, color=col, ms=4, mec="white", mew=0.5, zorder=3, label=nm if i == 0 else None)
    clean(cx, grid="y"); cx.set_xticks(range(len(COND))); cx.set_xticklabels(COND, fontsize=8)
    cx.set_ylabel("% of 200 replicates\n(CP 95% CI)")
    cx.set_title("c  Replicates enabling ≥1 track whose true gain is below m", loc="left")
    cx.legend(frameon=False, fontsize=7.2, loc="upper left")
    save(fig, "R47_Fig6_T2_selection")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

if __name__ == "__main__":
    for k, f in FIGS.items():
        f(); print("Fig", k, "ok")
