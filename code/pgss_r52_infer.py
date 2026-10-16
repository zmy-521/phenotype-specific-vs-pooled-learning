# -*- coding: utf-8 -*-
r"""R52 推断包装：**不改任何推断函数**，在独立进程中替换原推断脚本依赖的模块级引用后调用原 main()（与 pgss_r46_inference 同一做法）。
  ① pgss_r44_nhanes_run.DATA → 分支数据（new）或原冻结数据（zero）；
  ② task_arrays → O_CVD10 删除 v2 不可评价者、保留者取 v2 标签（R46 规则；对不含该人的任务是空操作）；
  ③ RUNS → 分支运行目录（new）或现行冻结单元的路由（zero：R46 v2 受影响单元 / 其余 R44、R45 原目录）；
  ④ 输出目录与目标推断用于逐位复核 E1 的 R44 推断位置。
模式
  new  ：分支敏感性推断 → 35_R52_.../inference/R44_INFERENCE.xlsx、target_inference/R45_TARGET_INFERENCE.xlsx
  zero ：零修正（原数据 + 现行冻结单元）→ zero_inference/…；check 要求与 R46 v2 冻结表逐格相同
  check：零修正 vs R46 v2 冻结推断表（全部比较表逐格相同），配变异对照
用法：python pgss_r52_infer.py new|zero|check  （new 与 zero 各在独立进程里运行，模块状态互不污染）
"""
import sys, json, pathlib, time, shutil
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r52_sens as S

OUTS = {"new": S.BR, "zero": S.BR / "zero_inference"}
V2 = {"nh44": S.R46 / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx",
      "target": S.R46 / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx"}
SKIP_SHEETS = {"0_汇总"}          # 含运行秒数


class Router:
    def __init__(self, base, alt, tags):
        self.base, self.alt, self.tags, self.hits = pathlib.Path(base), pathlib.Path(alt), set(tags), set()

    def __truediv__(self, tag):
        if tag in self.tags:
            self.hits.add(tag); return self.alt / tag
        return self.base / tag


def patch_arrays(NR):
    import pgss_xgb_engine as EN
    orig = NR.task_arrays

    def f(oc, ph):
        A, names, gs = orig(oc, ph)
        if oc != "O_CVD10":
            return A, names, gs
        ev, yv2 = S.cvd_status()
        keep = ev.reindex(A["uid"]).to_numpy(bool)
        lab = yv2.reindex(A["uid"]).to_numpy(float)
        y2 = A["y"].copy(); y2[keep] = lab[keep].astype(int)
        return EN.sub(dict(A, y=y2), np.flatnonzero(keep)), names, gs[keep]
    NR.task_arrays = f


def run(mode):
    import pgss_r44_nhanes_run as NR, pgss_r45_target as TG, pgss_r46_rerun as RR
    out = OUTS[mode]
    for sub in ("inference", "target_inference"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / "inference").mkdir(parents=True); (out / "target_inference").mkdir(parents=True)
    t0 = time.time(); reg = {}
    S.use_data(mode == "zero")
    patch_arrays(NR)
    if mode == "new":
        NR.RUNS = S.DIRS["main"]; TG.RUNS = S.DIRS["pop"]
    else:
        aff = set(RR.tasks("nh44"))
        NR.RUNS = Router(S.R44D / "runs", RR.DIRS["nh44"], [NR.tag_of(oc, ph, s, f) for (oc, ph) in aff for s in NR.SEEDS for f in range(NR.K_OUTER)])
        TG.RUNS = Router(S.R45T / "runs", RR.DIRS["nhpop"], [TG.tag_of("pop", oc, ph, s, f) for (oc, ph) in aff for s in NR.SEEDS for f in range(NR.K_OUTER)])
    NR.R44 = out
    import pgss_r44_nhanes_inference as NI
    NI.OUT = out / "inference"; NI.main()
    import pgss_r45_target_inference as TI
    shutil.copy2(S.R45T / "INCLUDED_TASKS.json", out / "target_inference" / "INCLUDED_TASKS.json")
    TI.OUT = out / "target_inference"; TI.main()
    for k, r in (("main", NR.RUNS), ("target", TG.RUNS)):
        if isinstance(r, Router):
            reg[k] = dict(路由到R46的tag数=len(r.tags), 实际命中=len(r.hits), 未命中=sorted(r.tags - r.hits))
        else:
            reg[k] = dict(运行目录=str(pathlib.Path(r).relative_to(S.ROOT)))
    rec = dict(模式=mode, 数据目录=str(pathlib.Path(NR.DATA).relative_to(S.ROOT)), 数据指纹=NR.data_fp(), 路由=reg, 秒=round(time.time() - t0))
    (out / f"_INFER_{mode}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    if any(v.get("未命中") for v in reg.values()):
        raise SystemExit(f"路由登记的 tag 未被读取：{reg}")
    print(json.dumps(rec, ensure_ascii=False))


def diff(a, b):
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


def check():
    rows = []
    pairs = {"nh44": OUTS["zero"] / "inference" / "R44_INFERENCE.xlsx", "target": OUTS["zero"] / "target_inference" / "R45_TARGET_INFERENCE.xlsx"}
    for k, zf in pairs.items():
        Z = pd.read_excel(zf, sheet_name=None); O = pd.read_excel(V2[k], sheet_name=None)
        rows.append(dict(推断=k, 表="（表名集合）", 行数=len(O), 不同格数=0 if set(Z) == set(O) else -1))
        for sh in O:
            if sh in SKIP_SHEETS:
                continue
            rows.append(dict(推断=k, 表=sh, 行数=len(O[sh]), 不同格数=diff(Z[sh], O[sh])))
    C = pd.DataFrame(rows)
    O = pd.read_excel(V2["nh44"], sheet_name="2_Brier对照"); m1 = O.copy(); m1.loc[0, "观察相对改善"] += 1e-12
    m2 = O.copy(); m2.loc[0, "三态_m0.010"] = "支持实质收益" if m2.loc[0, "三态_m0.010"] != "支持实质收益" else "排除实质收益"
    mut = [dict(变异="数值 +1e-12", 检出=diff(m1, O) > 0), dict(变异="三态改写", 检出=diff(m2, O) > 0)]
    ok = bool((C.不同格数 == 0).all() and all(m["检出"] for m in mut))
    rec = dict(全部通过=ok, 比较=C.to_dict("records"), 变异对照=mut, 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (S.BR / "R52_INFER_ZERO_CHECK.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(C.to_string()); print(mut, "\n全部通过:", ok)
    return ok


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "check":
        sys.exit(0 if check() else 1)
    run(m)
