# -*- coding: utf-8 -*-
r"""R46 修正后推断（方案 R46_OUTCOME_FIX_PROTOCOL_v1.0.md §1.4、§2、§3）。**不改任何推断函数**：
在独立进程中替换原推断脚本所依赖的三处模块级引用后调用原 main()：
  ① 任务数组 task_arrays —— 受影响任务删除不可评价人次（保留者标签取 v2）；
  ② 运行目录 RUNS —— 受影响单元路由到 R46 v2 目录，其余单元读原目录（Router，路由表登记在输出 json）；
  ③ 输出目录 OUT（以及目标推断复核所读的 R44 推断位置）。
模式
  nwicu / nh44 / target          ：修正后推断（target 依赖 nh44 的输出）
  nwicu-zero / nh44-zero / target-zero ：零修正（不路由、不删除），必须逐位复现原推断表（等价测试）
  check                          ：比较：零修正 vs 原表（全部相同）；修正后 vs 原表（未受影响任务的对照行全部相同）；配变异对照
用法：python pgss_r46_inference.py <模式>
"""
import sys, json, pathlib, time, shutil
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
R46 = ROOT / "28_R46_最终核验与T2"
OUTS = {"nwicu": R46 / "nwicu_v2" / "inference", "nh44": R46 / "nhanes_v2" / "r44v2" / "inference",
        "target": R46 / "nhanes_v2" / "target_v2"}
ZOUT = R46 / "outcome_fix" / "zero_inference"
ORIG = {"nwicu": ROOT / "27_R45_外部复现与机制核验" / "nwicu" / "inference" / "R45_NWICU_INFERENCE.xlsx",
        "nh44": ROOT / "25_R44_NHANES跨领域复现" / "inference" / "R44_INFERENCE.xlsx",
        "target": ROOT / "27_R45_外部复现与机制核验" / "target" / "R45_TARGET_INFERENCE.xlsx"}
FNAME = {"nwicu": "R45_NWICU_INFERENCE", "nh44": "R44_INFERENCE", "target": "R45_TARGET_INFERENCE"}
SHEETS = {"nwicu": ["2_Brier对照", "3_logloss对照"], "nh44": ["2_Brier对照", "3_logloss对照"],
          "target": ["2_Brier对照", "3_logloss对照"]}


class Router:
    """RUNS / tag：登记的受影响 tag → v2 目录，其余 → 原目录。"""
    def __init__(self, base, alt, tags):
        self.base, self.alt, self.tags, self.hits = pathlib.Path(base), pathlib.Path(alt), set(tags), set()

    def __truediv__(self, tag):
        if tag in self.tags:
            self.hits.add(tag); return self.alt / tag
        return self.base / tag


def patch_arrays(M, kind, zero):
    import pgss_xgb_engine as EN, pgss_r46_rerun as RR
    orig = M.task_arrays
    aff = set(RR.tasks(kind))

    def f(oc, ph):
        A, names, gs = orig(oc, ph)
        if zero or (oc, ph) not in aff:
            return A, names, gs
        ev, yv2, _ = RR.status(kind)
        keep = ev.reindex(A["uid"]).to_numpy(bool)
        lab = yv2.reindex(A["uid"]).to_numpy(float)
        y2 = A["y"].copy(); y2[keep] = lab[keep].astype(int)
        return EN.sub(dict(A, y=y2), np.flatnonzero(keep)), names, gs[keep]
    M.task_arrays = f


def route(M, kind, zero):
    import pgss_r46_rerun as RR
    tags = [] if zero else [RR.tag_of(*u) for u in RR.units(kind)]
    M.RUNS = Router(M.RUNS, RR.DIRS[kind], tags)
    return M.RUNS


def run(mode):
    zero = mode.endswith("-zero"); kind = mode.replace("-zero", "")
    out = (ZOUT / kind) if zero else OUTS[kind]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time(); routers = {}
    if kind == "nwicu":
        import pgss_r45_nwicu_run as NW, pgss_r45_nwicu_inference as NWI
        patch_arrays(NW, "nwicu", zero); routers["nwicu"] = route(NW, "nwicu", zero)
        NWI.OUT = out; NWI.main()
    elif kind == "nh44":
        import pgss_r44_nhanes_run as NR, pgss_r44_nhanes_inference as NI
        patch_arrays(NR, "nh44", zero); routers["nh44"] = route(NR, "nh44", zero)
        NI.OUT = out; NI.main()
    else:
        import pgss_r44_nhanes_run as NR, pgss_r45_target as TG, pgss_r45_target_inference as TI
        patch_arrays(NR, "nh44", zero); routers["nh44"] = route(NR, "nh44", zero); routers["nhpop"] = route(TG, "nhpop", zero)
        if not zero:
            NR.R44 = OUTS["nh44"].parent            # E1 逐位复核对照 v2 的 R44 推断
            assert (NR.R44 / "inference" / "R44_INFERENCE.xlsx").exists(), "先运行 nh44"
        shutil.copy2(TG.OUT / "INCLUDED_TASKS.json", out / "INCLUDED_TASKS.json")
        TI.OUT = out; TI.main()
    reg = {k: dict(路由到v2的tag数=len(r.tags), 实际命中=len(r.hits), 未命中=sorted(r.tags - r.hits)) for k, r in routers.items()}
    (out / "_ROUTING.json").write_text(json.dumps(dict(模式=mode, 路由=reg, 秒=round(time.time() - t0)), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    bad = [k for k, v in reg.items() if v["未命中"]]
    if bad:
        raise SystemExit(f"路由登记的 tag 未被读取：{bad}")
    print(json.dumps(reg, ensure_ascii=False))


# ── 比较 ─────────────────────────────────────────────────────────
def diff(a, b):
    """两表逐格比较（数值逐位、非数值字符串相等）；返回不同的格数。"""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return -1
    n = 0
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            n += int((~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))).sum())
        else:
            n += int((x.astype(str).to_numpy() != y.astype(str).to_numpy()).sum())
    return n


def unaffected_mask(kind, df):
    import pgss_r46_rerun as RR
    aff = set(RR.tasks("nhpop" if kind == "target" else kind))
    return ~df.apply(lambda r: (r.结局, r.表型) in aff, axis=1).to_numpy()


def check():
    rows = []
    for kind in ("nwicu", "nh44", "target"):
        for sh in SHEETS[kind]:
            o = pd.read_excel(ORIG[kind], sheet_name=sh)
            zf = ZOUT / kind / f"{FNAME[kind]}.xlsx"
            z = pd.read_excel(zf, sheet_name=sh)
            rows.append(dict(推断=kind, 表=sh, 比较="零修正 vs 原表（全部行）", 行数=len(o), 不同格数=diff(z, o)))
            v = pd.read_excel(OUTS[kind] / f"{FNAME[kind]}.xlsx", sheet_name=sh)
            mo, mv = unaffected_mask(kind, o), unaffected_mask(kind, v)
            rows.append(dict(推断=kind, 表=sh, 比较="修正后 vs 原表（未受影响任务的行）", 行数=int(mo.sum()),
                             不同格数=diff(v[mv].reset_index(drop=True), o[mo].reset_index(drop=True))))
            rows.append(dict(推断=kind, 表=sh, 比较="修正后 vs 原表（受影响任务的行，预期不同）", 行数=int((~mo).sum()),
                             不同格数=diff(v[~mv].reset_index(drop=True), o[~mo].reset_index(drop=True))))
    C = pd.DataFrame(rows)
    # 变异对照：原表一格 +1e-12 必须被 diff 检出；一行字符串改动必须检出
    o = pd.read_excel(ORIG["nwicu"], sheet_name="2_Brier对照"); m1 = o.copy()
    col = "观察相对改善"; m1.loc[0, col] = m1.loc[0, col] + 1e-12
    m2 = o.copy(); m2.loc[0, "三态_m0.010"] = "支持实质收益" if m2.loc[0, "三态_m0.010"] != "支持实质收益" else "排除实质收益"
    mut = [dict(变异="数值 +1e-12", 检出=diff(m1, o) > 0), dict(变异="三态改写", 检出=diff(m2, o) > 0)]
    ok = bool((C[C.比较.str.startswith("零修正")].不同格数 == 0).all()
              and (C[C.比较.str.contains("未受影响")].不同格数 == 0).all()
              and (C[C.比较.str.contains("预期不同")].不同格数 != 0).all() and all(m["检出"] for m in mut))
    summ = dict(全部通过=ok, 比较=C.to_dict("records"), 变异对照=mut, 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (R46 / "outcome_fix" / "R46_INFERENCE_CHECK.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str),
                                                                  encoding="utf-8")
    print(C.to_string()); print(mut, "\n全部通过:", ok)
    return ok


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "check":
        sys.exit(0 if check() else 1)
    run(m)
