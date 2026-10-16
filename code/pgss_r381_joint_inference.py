# -*- coding: utf-8 -*-
r"""R38.1 §4：**联合** cluster bootstrap —— 修正 R38 §C 的下标对应错误。

R38 §C 错在哪
    `task_stats` 对每个任务独立建 cluster 集合与 rng，总体摘要再把各任务的重抽向量
    按下标 vstack 平均（并用 `m[:n]` 截短对齐）。**同 seed 不等于同一患者获得同一权重**：
    各任务可评估人群不同（本项目实测 cluster 数 45,888–94,920 不等），
    `rng.choice(uq)` 抽的是不同的 uq。GPT 的反例在本机精确复现：
    任务 A{p1,p2,p3} 与 B{p2,p3,p4} 同 seed=11，p2 在 2000 次中 **1473 次权重不同**。
    ⇒ **逐任务区间仍有效；库内与两库总体区间无效，必须重算。**

本脚本的做法
    1. 按库建**以真实 ID 为键**的统一 cluster 注册表（MIMIC=患者；eICU=冻结的患者—医院连通分量）
    2. 每个 `replicate_id` 只生成**一套**权重（对注册表做有放回抽样 → 多重度向量 m）
    3. 同一套 m 投影到该库**全部任务与两个候选**；任务只用自己名下的单位
    4. 在同一次重抽内依次算：逐轨风险 → 逐任务 Δ/相对改善 → 库内 12 任务等权 → 两库等权
    5. 两库使用**独立随机流**（不同 SeedSequence 子流），不共享权重
    6. 分母（R_SHARED+）在每次重抽内重新计算

实现要点（既正确又可算）
    把每个任务按注册表槽位预聚合成逐 cluster 的和与计数：
        nA, nB, sumRefA, sumRefB, sumCandA, sumCandB   （不在该任务的 cluster 填 0）
    则一次重抽的风险是对多重度向量的加权比：
        RrefA = (m·sumRefA)/(m·nA)
    于是 B 次重抽 = 一次 (M×B) 多重度矩阵与各任务 6 条向量的矩阵乘，分块以控内存。
    **这不是近似，与逐次展开索引数学上等价**，只是避免了 2.4e9 次元素展开。

推断范围（**必须随结论一起写出，不得省略**）
    本区间为 `fixed-OOF-prediction cluster-resampling intervals`：
    它只刻画**已完成的那组 OOF 预测**在患者/医院簇层面的抽样不确定性，
    **不包含重新训练与重新选择模型带来的不确定性**。交叉验证训练集重叠会产生相关性，
    仅靠患者分组隔离**不足以保证算法级区间的覆盖率**（见 Bates/Hastie/Tibshirani
    关于交叉验证误差推断的讨论，arXiv:2104.00673）。
    ⇒ **不得**升级为「重新开发这些算法时真实收益已被 95% 把握排除在 1% 以上」。

分类改为两个维度（GPT 第 39 轮）
    维度一 `upper_below_margin`：95% CI 上界是否 < 1%
    维度二 `ci_direction`：区间跨零 / 全正 / 全负
    旧 A/C/D 把两件事混在一列，会让人误读成「只有 A 类才满足上界 <1%」。
    方向判断为**探索性、未做多重发现校正**，不得确认某表型必然获益或有害。

用法： python pgss_r381_joint_inference.py [--B 2000]
"""
import json, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).parent))
import pgss_threads  # noqa: F401  必须在 numpy 之前
import pgss_dataver as _DV
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
O = ROOT / "19_R38"; PRED = O / "bench" / "preds"
B_BOOT = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 2000
MARGIN = 0.01
MARGIN_NAME = "prespecified material-gain margin"
CAND = ["FR-PP", "Hard-both"]
REF = "SHARED+"
CHUNK = 200          # 每块的重抽数，控内存
RNG_ROOT = 20381     # 顶层种子；两库各取一条独立子流


class PredError(RuntimeError):
    """预测文件的身份/完整性检查失败。**必须抛出，不得静默修复。**

    R38 用 `fs[0]`、`groupby().mean()`、`meta.first()` 读预测，
    会把「重复预测、缺 seed、ID 集合不一致、标签不一致」静默掩盖。
    """


def load_task_strict(db, oc, ph, seeds):
    """严格读取一个任务的三 seed 预测，任何不一致即抛 PredError。

    检查项（GPT 第 39 轮指定）
        每 seed 恰一个文件 · 每单位恰三个 seed · ID 集合跨 seed 一致
        结局与表型标签跨 seed 一致 · 无重复单位 · 无缺失预测
    """
    need = [REF] + CAND
    per = {}
    for sd in seeds:
        fs = sorted(PRED.glob(f"{db}__{oc}__{ph}__{sd}__*preds.parquet"))
        if len(fs) != 1:
            raise PredError(f"{db}/{oc}/{ph} seed={sd} 命中 {len(fs)} 个预测文件（应为 1）")
        d = pd.read_parquet(fs[0])
        miss = [c for c in need if c not in d.columns]
        if miss: raise PredError(f"{fs[0].name} 缺程序列 {miss}")
        if d.unit_id.duplicated().any():
            raise PredError(f"{fs[0].name} 有 {int(d.unit_id.duplicated().sum())} 个重复单位")
        if d[need].isna().any().any():
            raise PredError(f"{fs[0].name} 存在缺失预测")
        per[sd] = d.set_index("unit_id").sort_index()
    ids = [set(v.index) for v in per.values()]
    if len(set(map(frozenset, ids))) != 1:
        a, b = ids[0], ids[1]
        raise PredError(f"{db}/{oc}/{ph} 跨 seed 的单位集合不一致"
                        f"（对称差 {len(a ^ b)}）")
    base = per[seeds[0]]
    # 轨道列在 bench 里写作 `subgroup`（"A"/"B"），不是 `z`
    for sd in seeds[1:]:
        for col in ("y", "subgroup"):
            if not np.array_equal(base[col].to_numpy(), per[sd][col].to_numpy()):
                raise PredError(f"{db}/{oc}/{ph} seed={sd} 的 {col} 与 seed={seeds[0]} 不一致")
    return per, base


def track_mask(df):
    """A 轨布尔掩码。预测文件用 subgroup ∈ {"A","B"} 表示轨道。"""
    return (df["subgroup"].astype(str).to_numpy() == "A")


def brier_terms(per, base, seeds):
    """逐单位、对三 seed 取平均的 Brier 损失（三 seed 不是 3 倍样本量）"""
    y = base.y.to_numpy(float)
    out = {}
    for prog in [REF] + CAND:
        L = np.mean([(per[sd][prog].to_numpy(float) - y) ** 2 for sd in seeds], axis=0)
        out[prog] = L
    return out, track_mask(base)


def global_cluster_map(db):
    """返回 {患者ID -> 全局 cluster ID}，在**全部合格人群**上算一次。

    为什么不能按任务算
        `pgss_r38_bench.joint_groups` 返回的是 union-find 的**根标签**，
        取值依赖行序与哪些行在场。不同表型的可评估人群不同 ⇒
        同一患者在不同任务里可能拿到不同标签，且分量会因缺少「连接患者」而裂开
        （实测：单任务 64 个分量，12 任务并集却出现 101 个标签）。
        于是「统一注册表」就名存实亡 —— 跨任务按标签对齐等于对齐了不同的东西。

    为什么全局分量是对的，也是保守的
        依赖结构（哪些患者共享医院）是**数据的性质**，不随任务变。
        行越多分量只会越粗（合并不会拆分），因而全局分量对依赖的处理更保守。

    注意：外层划分仍按**任务内**分量进行（那是「测试集不得与 dev 共享医院/患者」
    的要求，任务内满足即可），本函数只用于跨任务的联合重抽。
    """
    import pgss_r38_bench as BN
    X, P_, OK, meta = BN.load(db)
    e = meta.eligible.fillna(False).to_numpy(bool)
    pid = meta.group_id.astype(str).to_numpy()[e]
    if db == "mimic":
        return {p: p for p in np.unique(pid)}          # MIMIC 以患者为 cluster，本就全局稳定
    hid = meta.hospital_id.astype(str).to_numpy()[e]
    comp = BN.joint_groups(pid, hid)
    return dict(zip(pid, comp))


def kish_effective(counts):
    """Kish 有效 cluster 数 1/Σpᵢ²。远小于 cluster 数时说明重抽被少数大簇主导。"""
    c = np.asarray(counts, float); p = c / c.sum()
    return float(1.0 / np.sum(p ** 2))


def main():
    t0 = time.time()
    import pgss_r38_bench as BN
    seeds = list(BN.SEEDS)
    # ── 一、按库建统一注册表并预聚合 ──
    DB = {}; kish_by_db = {}
    for db in ("mimic", "eicu"):
        gmap = global_cluster_map(db)
        print(f"  {db} 全局 cluster 映射：{len(set(gmap.values()))} 个分量"
              f"（覆盖 {len(gmap)} 个患者）  {time.time()-t0:.0f}s", flush=True)
        tasks, reg_keys = {}, set()
        for oc in BN.OUTCOMES:
            for ph in BN.PHENOS:
                per, base = load_task_strict(db, oc, ph, seeds)
                L, z = brier_terms(per, base, seeds)
                gid = base.group_id.astype(str).to_numpy()
                miss = [p for p in np.unique(gid) if p not in gmap]
                if miss:
                    raise PredError(f"{db}/{oc}/{ph} 有 {len(miss)} 个患者不在全局映射中"
                                    f"（示例 {miss[:3]}）")
                cl = np.array([gmap[p] for p in gid], dtype=object)
                tasks[(oc, ph)] = dict(L=L, z=z, cl=cl)
                reg_keys |= set(np.unique(cl))
                print(f"  读 {db}/{oc}/{ph}  n={len(z)}  cluster={len(set(cl))}"
                      f"  {time.time()-t0:.0f}s", flush=True)
        reg = np.array(sorted(reg_keys), dtype=object)
        slot = {k: i for i, k in enumerate(reg)}
        M = len(reg)
        for key, T in tasks.items():
            s = np.fromiter((slot[c] for c in T["cl"]), dtype=np.int64, count=len(T["cl"]))
            z = T["z"]; agg = {}
            for trk, m in (("A", z), ("B", ~z)):
                agg[f"n{trk}"] = np.bincount(s[m], minlength=M).astype(float)
                for prog in [REF] + CAND:
                    agg[f"{prog}|{trk}"] = np.bincount(s[m], weights=T["L"][prog][m],
                                                       minlength=M)
            T["agg"] = agg; T["slot"] = s
        DB[db] = dict(tasks=tasks, reg=reg, M=M)
        _tot = np.zeros(M)
        for T in tasks.values():
            _tot += T["agg"]["nA"] + T["agg"]["nB"]
        kish_by_db[db] = round(kish_effective(_tot[_tot > 0]), 1)
        print(f"  {db} Kish 有效分量数 = {kish_by_db[db]}（名义 {M}）", flush=True)
        print(f"  {db} 统一注册表 M={M}（12 任务并集）  {time.time()-t0:.0f}s", flush=True)

    # ── 二、联合重抽：每 replicate 一套权重，投影到全部任务 ──
    ss = np.random.SeedSequence(RNG_ROOT).spawn(2)   # 两库独立随机流
    streams = dict(zip(("mimic", "eicu"), ss))
    per_task_boot = {}     # (db,oc,ph,cand) -> rel 向量（按 replicate_id 对齐）
    per_task_abs = {}
    db_mean_boot = {}      # (db,cand) -> 库内 12 任务等权的 rel 向量

    def risks(aggs, m):
        """给定多重度向量/矩阵 m（M 或 M×b），返回各程序两轨风险"""
        out = {}
        for trk in ("A", "B"):
            den = aggs[f"n{trk}"] @ m
            for prog in [REF] + CAND:
                out[f"{prog}|{trk}"] = np.where(den > 0, (aggs[f"{prog}|{trk}"] @ m) / den, np.nan)
            out[f"den|{trk}"] = den
        return out

    for db, pack in DB.items():
        M = pack["M"]; rng = np.random.default_rng(streams[db])
        keys = sorted(pack["tasks"].keys())
        acc = {(k, c): [] for k in keys for c in CAND}
        acc_abs = {(k, c): [] for k in keys for c in CAND}
        dbm = {c: [] for c in CAND}
        done = 0
        while done < B_BOOT:
            b = min(CHUNK, B_BOOT - done)
            # 一次性抽 b 个 replicate 的多重度矩阵 (M×b)
            draw = rng.integers(0, M, size=(M, b))
            Mt = np.zeros((M, b))
            for j in range(b):
                Mt[:, j] = np.bincount(draw[:, j], minlength=M)
            rel_stack = {c: [] for c in CAND}
            for k in keys:
                aggs = pack["tasks"][k]["agg"]
                R = risks(aggs, Mt)
                okm = (R["den|A"] > 0) & (R["den|B"] > 0)
                Rref = .5 * (R[f"{REF}|A"] + R[f"{REF}|B"])
                for c in CAND:
                    Rc = .5 * (R[f"{c}|A"] + R[f"{c}|B"])
                    dA = R[f"{REF}|A"] - R[f"{c}|A"]; dB = R[f"{REF}|B"] - R[f"{c}|B"]
                    de = .5 * (dA + dB)
                    rel = np.where((Rref > 0) & okm, (Rref - Rc) / Rref, np.nan)
                    acc[(k, c)].append(rel); acc_abs[(k, c)].append(de)
                    rel_stack[c].append(rel)
            for c in CAND:
                dbm[c].append(np.nanmean(np.vstack(rel_stack[c]), axis=0))
            done += b
            print(f"  {db} 联合重抽 {done}/{B_BOOT}  {time.time()-t0:.0f}s", flush=True)
        for (k, c), v in acc.items():
            per_task_boot[(db, k[0], k[1], c)] = np.concatenate(v)
            per_task_abs[(db, k[0], k[1], c)] = np.concatenate(acc_abs[(k, c)])
        for c in CAND: db_mean_boot[(db, c)] = np.concatenate(dbm[c])

    # ── 三、观察值（原样本，非重抽均值）──
    rows = []
    for db, pack in DB.items():
        for (oc, ph), T in sorted(pack["tasks"].items()):
            one = np.ones(pack["M"])
            R = risks(T["agg"], one)
            Rref = .5 * (R[f"{REF}|A"] + R[f"{REF}|B"])
            for c in CAND:
                Rc = .5 * (R[f"{c}|A"] + R[f"{c}|B"])
                dA = float(R[f"{REF}|A"] - R[f"{c}|A"]); dB = float(R[f"{REF}|B"] - R[f"{c}|B"])
                rb = per_task_boot[(db, oc, ph, c)]; ab = per_task_abs[(db, oc, ph, c)]
                lo, hi = (float(np.nanquantile(rb, .025)), float(np.nanquantile(rb, .975)))
                upper_below = bool(hi < MARGIN)
                direction = ("全负" if hi < 0 else "全正" if lo > 0 else "跨零")
                _cnt = (T["agg"]["nA"] + T["agg"]["nB"])
                _cnt = _cnt[_cnt > 0]
                rows.append(dict(
                    database=db, outcome=oc, phenotype=ph, candidate=c,
                    n_units=int(len(T["z"])), n_clusters=int(len(_cnt)),
                    n_clusters_effective_kish=round(kish_effective(_cnt), 1),
                    registry_M=int(pack["M"]),
                    observed_d_A=dA, observed_d_B=dB, observed_d_equal=.5 * (dA + dB),
                    observed_relative_gain=float((Rref - Rc) / Rref),
                    rel_P2_5=lo, rel_P97_5=hi,
                    abs_P2_5=float(np.nanquantile(ab, .025)),
                    abs_P97_5=float(np.nanquantile(ab, .975)),
                    upper_below_margin=upper_below, ci_direction=direction,
                    R_ref=float(Rref), R_cand=float(Rc), B=int(np.isfinite(rb).sum())))
    R = pd.DataFrame(rows)

    # ── 四、总体摘要：库内 → 两库，均在同一 replicate 内 ──
    summ = []
    for c in CAND:
        vs = {}
        for db in ("mimic", "eicu"):
            v = db_mean_boot[(db, c)]; vs[db] = v
            sub = R[(R.database == db) & (R.candidate == c)]
            summ.append(dict(范围=f"{db} 12 任务等权", candidate=c,
                             observed_task_mean=float(sub.observed_relative_gain.mean()),
                             P2_5=float(np.nanquantile(v, .025)),
                             P97_5=float(np.nanquantile(v, .975)),
                             upper_below_margin=bool(np.nanquantile(v, .975) < MARGIN),
                             B=int(np.isfinite(v).sum())))
        n = min(len(v) for v in vs.values())
        both = .5 * (vs["mimic"][:n] + vs["eicu"][:n])   # 独立流，按 replicate_id 配对
        sub = R[R.candidate == c]
        summ.append(dict(范围="两库等权", candidate=c,
                         observed_task_mean=float(sub.observed_relative_gain.mean()),
                         P2_5=float(np.nanquantile(both, .025)),
                         P97_5=float(np.nanquantile(both, .975)),
                         upper_below_margin=bool(np.nanquantile(both, .975) < MARGIN),
                         B=int(np.isfinite(both).sum())))
    S = pd.DataFrame(summ)

    # ── 五、与 R38 旧（无效）总体区间的对照，证明差异真实存在 ──
    cmpr = []
    old = O / "_archive_r38_negative_inference.parquet"
    if old.exists():
        Oldr = pd.read_parquet(old)
        for _, r in S.iterrows():
            if r["范围"] == "两库等权": continue
            dbn = r["范围"].split()[0]
            o = Oldr[(Oldr.database == dbn) & (Oldr.candidate == r.candidate)]
            if len(o):
                cmpr.append(dict(范围=r["范围"], candidate=r.candidate,
                                 新_P2_5=r.P2_5, 新_P97_5=r.P97_5,
                                 旧逐任务端点平均_P2_5=float(o.rel_P2_5.mean()),
                                 旧逐任务端点平均_P97_5=float(o.rel_P97_5.mean()),
                                 说明="旧值由逐任务端点平均而来，非联合重抽，仅作差异展示"))
    CMP = pd.DataFrame(cmpr)

    stamp = dict(
        预测目录=str(PRED), 预测文件数=len(list(PRED.glob("*preds.parquet"))),
        数据版本=(json.loads((_DV.DIR / "DATA_VERSION.json").read_text(encoding="utf-8"))
                 .get("数据版本") if (_DV.DIR / "DATA_VERSION.json").exists() else "未知"),
        B=B_BOOT, 任务数=int(R.groupby(["database", "outcome", "phenotype"]).ngroups),
        候选=CAND, margin=MARGIN, margin_名称=MARGIN_NAME,
        重抽方案="joint-cluster-bootstrap/unified-registry/shared-weights-per-replicate",
        推断范围="fixed-OOF-prediction cluster-resampling intervals",
        注册表规模={db: int(p["M"]) for db, p in DB.items()},
        有效分量数_Kish=kish_by_db,
        cluster口径="MIMIC=患者；eICU=在全部合格人群上一次性算出的患者—医院连通分量（全局，非任务内）",
        随机流=f"SeedSequence({RNG_ROOT}).spawn(2) 两库独立",
        时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (O / "R381_JOINT_INFERENCE_STAMP.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")
    R.to_parquet(O / "R381_JOINT_INFERENCE.parquet", index=False)

    note = pd.DataFrame([
        dict(项="来源戳", 值=json.dumps(stamp, ensure_ascii=False)),
        dict(项="推断范围", 值=("fixed-OOF-prediction cluster-resampling intervals："
                            "只刻画已完成 OOF 预测在簇层面的抽样不确定性，"
                            "**不含重新训练与模型选择的不确定性**；交叉验证训练集重叠产生相关性，"
                            "仅靠患者分组不足以保证算法级覆盖率（arXiv:2104.00673）。"
                            "不得表述为「重新开发算法时真实收益已被排除在 1% 以上」。")),
        dict(项="margin 口径", 值=f"{MARGIN:.0%} 为 {MARGIN_NAME}，**不是 MCID**，未经临床效用验证"),
        dict(项="分类维度", 值="upper_below_margin（上界是否<1%）与 ci_direction（跨零/全正/全负）分列；"
                            "方向判断为探索性、未做多重发现校正"),
        dict(项="**重要限制：有效分量数**",
             值=("eICU 按患者—医院连通分量聚类后，分量数远少于就诊数："
                 "109,079 例就诊 / 94,901 患者 / 207 医院 ⇒ 64 个分量，"
                 "**Kish 有效分量数约 22**（最大分量占 10.1%，前三占 26.0%）。"
                 "⇒ eICU 的区间实质上由 20 余个独立单位决定，"
                 "百分位 bootstrap 在这种有效样本量下覆盖率可能不佳，"
                 "**eICU 的「上界 <1%」比 MIMIC 弱得多**。"
                 "MIMIC 按患者聚类，有效数约 37,600，无此问题。")),
        dict(项="cluster 口径", 值="eICU 分量在**全部合格人群**上算一次（全局 ID）。"
                              "按任务分别算会使同一患者在不同任务拿到不同标签、"
                              "分量因缺少连接患者而裂开（实测 64 vs 并集 101 个标签），"
                              "统一注册表即失效。外层划分仍按任务内分量，两者用途不同。"),
        dict(项="修正内容", 值="R38 §C 各任务独立重抽后按下标平均无效（同 seed≠同患者权重，"
                            "实测 p2 在 2000 次中 1473 次权重不同）；本版按库统一注册表、"
                            "每 replicate 一套权重投影到全部任务"),
    ])
    with pd.ExcelWriter(O / "R381_JOINT_INFERENCE.xlsx", engine="openpyxl") as xw:
        note.to_excel(xw, sheet_name="0_口径与推断范围", index=False)
        R.round(8).to_excel(xw, sheet_name="1_逐任务CI与两维分类", index=False)
        S.round(8).to_excel(xw, sheet_name="2_总体摘要_联合重抽", index=False)
        if len(CMP): CMP.round(8).to_excel(xw, sheet_name="3_与旧无效区间对照", index=False)
        pd.crosstab(R.candidate, R.upper_below_margin).to_excel(xw, sheet_name="4_上界是否低于margin")
        pd.crosstab(R.candidate, R.ci_direction).to_excel(xw, sheet_name="5_区间方向")

    print("\n=== 上界是否 < margin（1%）===")
    print(pd.crosstab(R.candidate, R.upper_below_margin).to_string())
    print("\n=== 区间方向（探索性，未校正多重发现）===")
    print(pd.crosstab(R.candidate, R.ci_direction).to_string())
    print("\n=== 总体摘要（联合重抽）===")
    print(S.round(6).to_string(index=False))
    if len(CMP):
        print("\n=== 与旧「端点平均」区间的差异 ===")
        print(CMP.round(6).to_string(index=False))
    print(f"\n已写 {O/'R381_JOINT_INFERENCE.xlsx'}   {time.time()-t0:.0f}s")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
