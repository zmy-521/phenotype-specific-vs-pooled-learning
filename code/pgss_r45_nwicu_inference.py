# -*- coding: utf-8 -*-
r"""R45 NWICU 独立程序复现：统计汇总（与 R42/R44 同一推断口径；只读 nwicu/runs，不拟合）。方案：R45_NWICU_PROTOCOL_v1.0.md。

1. 任务纳入：该任务 3 seed × 5 外层折的 main 单元全部完成才纳入；否则整任务列为「不可评价」并写明原因（不换种子补跑）。
2. 逐单位损失：每个 seed 五折拼接成全队列 OOF，Brier 与 log loss 对 3 个 seed 取均值（单位集合 = 任务全体，断言）。
3. 臂：与 R42 相同（R42I.arm_columns）。
4. 对照：R42 的 E1–E7（不含认证）+ HTFS vs S0*（Brier）；E1、E2、HTFS vs S0*（log loss）。尺度：逐轨、两轨等权、人口加权。
5. 联合簇重抽：簇 = 患者（subject_id），全部纳入任务共用一个注册表，B = 2000，SeedSequence(20451)；百分位区间；三态（m ∈ {0.5%,1%,2%}）。
6. max-T：θ_tg = (1−m)R_tg(P*) − R_tg(HTFS)，纳入任务 × 2 轨，按重抽 SD 标准化。
7. 辅助：各 seed 完整 OOF 的 AUROC、AUPRC、校准截距/斜率（P*、HTFS、Hfull、S0*、SZ*），与逐 seed 配对差（原概率，不重校准）。
8. 开发选择：P* 选择分布、HTFS 的 (kA, kB) 分布。
推断范围：固定 OOF 预测的簇重抽（第 Ⅱ 层），不是算法级区间。
输出：27_R45_外部复现与机制核验/nwicu/inference/R45_NWICU_INFERENCE.xlsx、.json
"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r42_inference as R42I
import pgss_r43_readonly as RO
import pgss_r45_nwicu_run as NW

OUT = NW.NW / "inference"
DB = "nwicu"
B_BOOT, CHUNK, ROOT_SEED = 2000, 100, 20451
CONTRASTS = [c for c in R42I.CONTRASTS if not c[0].startswith("CERT")] + [("E1_vs_S0*", "S0*", "HTFS")]
LL_CONTRASTS = [("E1", "P*", "HTFS"), ("E2", "P*", "Hfull"), ("E1_vs_S0*", "S0*", "HTFS")]
LL_ARMS = ["P*", "HTFS", "Hfull", "S0*"]


def collect(oc, ph):
    per_seed, miss, choice = {}, [], []
    for seed in NW.SEEDS:
        parts = []
        for fold in range(NW.K_OUTER):
            d = NW.RUNS / NW.tag_of(oc, ph, seed, fold)
            if not (d / "_DONE").exists():
                why = "未运行"
                if (d / "_FAILED").exists():
                    try:
                        j = json.loads((d / "_FAILED").read_text(encoding="utf-8")); why = f"{j.get('error')} @ {j.get('last_role')}"
                    except Exception:
                        why = (d / "_FAILED").read_text(encoding="utf-8")[:200]
                miss.append(f"seed{seed} 折{fold}：{why}"); continue
            P = pd.read_parquet(d / "preds_test.parquet")
            sp = json.loads((d / "specs.json").read_text(encoding="utf-8"))
            parts.append(pd.DataFrame({"unit_id": P.unit_id, "y": P.y, "z": P.z, **R42I.arm_columns(P, sp)}))
            h = sp["specs"]["HTFS"]
            choice.append(dict(结局=oc, 表型=ph, seed=seed, 外层折=fold,
                               Pstar=f"{sp['pstar']}|{'cal' if sp['specs'][sp['pstar']]['cal'] else 'raw'}",
                               HTFS_kA=h["kA"], HTFS_kB=h["kB"]))
        if len(parts) == NW.K_OUTER:
            per_seed[seed] = pd.concat(parts).set_index("unit_id").sort_index()
    return per_seed, miss, choice


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tasks, status, AUX, PRS, CH = {}, [], [], [], []
    for oc in NW.OUTCOMES:
        for ph in NW.PHENOS:
            A, names, gs = NW.task_arrays(oc, ph)
            per_seed, miss, choice = collect(oc, ph)
            CH += choice
            ok = len(per_seed) == 3 and not miss
            if ok:
                idx = per_seed[NW.SEEDS[0]].index
                for s in NW.SEEDS:
                    assert per_seed[s].index.equals(idx), "各 seed 单位集合不同"
                assert np.array_equal(idx.to_numpy(np.int64), np.sort(A["uid"])), "OOF 单位集合 ≠ 任务全体"
            status.append(dict(结局=oc, 表型=ph, 任务人次=len(A["uid"]), 唯一患者=int(len(np.unique(gs))), A轨人次=int(A["z"].sum()),
                               事件=int(A["y"].sum()), A轨事件=int(A["y"][A["z"]].sum()), 纳入=ok, 缺失单元=len(miss),
                               原因="；".join(miss[:5])))
            if not ok:
                continue
            y = per_seed[NW.SEEDS[0]].y.to_numpy(float); z = per_seed[NW.SEEDS[0]].z.to_numpy(bool)
            arms = [c for c in per_seed[NW.SEEDS[0]].columns if c not in ("y", "z")]
            L = {a: np.mean([(d[a].to_numpy(float) - y) ** 2 for d in per_seed.values()], axis=0) for a in arms}
            LL = {a: np.mean([RO.logloss_vec(y, d[a].to_numpy(float)) for d in per_seed.values()], axis=0) for a in LL_ARMS}
            # 簇 = 患者：把单位映射到患者编码
            gmap = pd.Series(gs, index=A["uid"])
            cl = gmap.loc[idx.to_numpy(np.int64)].to_numpy()
            tasks[(oc, ph)] = (idx.to_numpy(np.int64), y, z, L, LL, cl)
            a_, p_ = RO.aux_rows(DB, oc, ph, {s: d[["y", "z", "P*", "HTFS", "Hfull", "S0*", "SZ*"]] for s, d in per_seed.items()},
                                 idx[:0])
            AUX += a_; PRS += p_
            print(f"{oc} {ph} 读入 {time.time()-t0:.0f}s", flush=True)
    ST = pd.DataFrame(status)
    keys = list(tasks)
    reg = np.unique(np.concatenate([tasks[k][5] for k in keys])); Mn = len(reg)
    agg = {}
    for k in keys:
        uid, y, z, L, LL, cl = tasks[k]
        s = np.searchsorted(reg, cl); a = {}
        for met, Ls in (("brier", L), ("logloss", LL)):
            for arm, v in Ls.items():
                for sc, msk in (("A", z), ("B", ~z)):
                    a[(met, arm, sc, "n")] = np.bincount(s[msk], minlength=Mn).astype(float)
                    a[(met, arm, sc, "s")] = np.bincount(s[msk], weights=v[msk], minlength=Mn)
        agg[k] = a
    rng = np.random.default_rng(np.random.SeedSequence(ROOT_SEED)); done = 0; reps = {}
    while done < B_BOOT:
        b = min(CHUNK, B_BOOT - done)
        draw = rng.integers(0, Mn, size=(Mn, b)); Mt = np.zeros((Mn, b))
        for j in range(b):
            Mt[:, j] = np.bincount(draw[:, j], minlength=Mn)
        for k in keys:
            a = agg[k]
            for (met, arm, sc, q) in [x for x in a if x[3] == "n" and x[2] == "A"]:
                nA = a[(met, arm, "A", "n")] @ Mt; sA = a[(met, arm, "A", "s")] @ Mt
                nB = a[(met, arm, "B", "n")] @ Mt; sB = a[(met, arm, "B", "s")] @ Mt
                for sc_, num, den in (("A", sA, nA), ("B", sB, nB), ("P", sA + sB, nA + nB)):
                    reps.setdefault((DB,) + k + (met, arm, sc_), []).append(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))
        done += b
        print(f"  重抽 {done}/{B_BOOT} {time.time()-t0:.0f}s", flush=True)
    reps = {k: np.concatenate(v) for k, v in reps.items()}
    T3 = {(DB,) + k: v[:5] for k, v in tasks.items()}
    T_b = RO.contrast_table(T3, reps, CONTRASTS, "brier")
    T_l = RO.contrast_table(T3, reps, LL_CONTRASTS, "logloss")
    TH, c95 = RO.maxT(T3, reps, "brier")
    dirs = []
    for met, d in (("Brier", T_b), ("logloss", T_l)):
        for (c, sc), h in d.groupby(["对照", "尺度"]):
            vc = h.相对0方向.value_counts(); t1 = h["三态_m0.010"].value_counts()
            dirs.append(dict(指标=met, 对照=c, 尺度=sc, 任务=len(h), 相对改善中位=h.观察相对改善.median(),
                             相对改善最小=h.观察相对改善.min(), 相对改善最大=h.观察相对改善.max(), 绝对差中位=h.观察绝对差.median(),
                             相对上界最大=h.相对_P97_5.max(), 全负=int(vc.get("全负", 0)), 跨零=int(vc.get("跨零", 0)),
                             全正=int(vc.get("全正", 0)), 排除m1=int(t1.get("排除实质收益", 0)),
                             证据不足m1=int(t1.get("证据不足", 0)), 支持m1=int(t1.get("支持实质收益", 0))))
    DIR = pd.DataFrame(dirs)
    CHD = pd.DataFrame(CH)
    CHS = pd.concat([CHD.groupby("Pstar").size().rename("单元数").reset_index().assign(项="P*选择"),
                     CHD.assign(k=lambda x: "kA=" + x.HTFS_kA.astype(str) + ",kB=" + x.HTFS_kB.astype(str))
                        .groupby("k").size().rename("单元数").reset_index().rename(columns={"k": "Pstar"}).assign(项="HTFS面板")])
    summ = dict(纳入任务=len(tasks), 任务总数=len(ST), 簇数=int(Mn), 簇定义="患者 subject_id", B=B_BOOT, maxT临界值=c95,
                maxT项数=len(TH), 秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R45_NWICU_INFERENCE.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        ST.to_excel(xw, sheet_name="1_任务纳入", index=False)
        T_b.to_excel(xw, sheet_name="2_Brier对照", index=False)
        T_l.to_excel(xw, sheet_name="3_logloss对照", index=False)
        DIR.to_excel(xw, sheet_name="4_对照方向汇总", index=False)
        TH.to_excel(xw, sheet_name="5_E1_maxT", index=False)
        pd.DataFrame(AUX).to_excel(xw, sheet_name="6_辅助指标逐seed", index=False)
        pd.DataFrame(PRS).to_excel(xw, sheet_name="7_配对差逐seed", index=False)
        CHD.to_excel(xw, sheet_name="8_开发选择逐单元", index=False)
        CHS.to_excel(xw, sheet_name="9_开发选择分布", index=False)
    (OUT / "R45_NWICU_INFERENCE.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
