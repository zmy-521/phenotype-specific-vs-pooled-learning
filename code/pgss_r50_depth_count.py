# -*- coding: utf-8 -*-
r"""R50（GPT–Claude v0.2 联合讨论第 1 轮 O5）：P* 实际选用树深 7 的频率——一次限定范围的只读配置计数。

范围（GPT 授权原文的执行要求）：只读 MIMIC-IV / eICU-CRD、main 分支、完整外层开发集 D 的已完成逐单元配置（23_R42_正式主实验/runs/*__main/specs.json）；
按保存的 pstar 键取最终选中的合并规格，读其真实 theta.max_depth；保留 family、weight、raw/cal 元数据。
不调用 develop()/fit()/predict()/再校准；不做删除深度 7 的反事实；不按结果改动冻结 P*。
核对：同一批记录得到的 P* 族|权重|校准 分布必须与已交付的 R42 证据表 C8_开发选择分布（main，P* 行）逐项相同——证明读到的是正确的记录。
附带（同一文件中的同类元数据，供判断专属网格上限是否常被触及）：HTFS 两轨所选深度。
输出：32_R50_讨论只读计数/（CSV、JSON、本脚本的 SHA）。
"""
import sys, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

ROOT = HERE.parent
RUNS = ROOT / "23_R42_正式主实验" / "runs"
C8 = ROOT / "23_R42_正式主实验" / "R42_正式主实验_证据表.xlsx"
OUT = ROOT / "32_R50_讨论只读计数"
DB = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD"}


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def read_units():
    rows, missing = [], []
    for d in sorted(RUNS.glob("*__main")):
        db, oc, ph, seed, fold, br = d.name.split("__")
        if db not in DB:
            continue
        f = d / "specs.json"
        if not (d / "_DONE").exists() or not f.exists():
            missing.append(dict(单元=d.name, 原因="缺 _DONE 或 specs.json")); continue
        j = json.loads(f.read_text(encoding="utf-8"))
        ps = j.get("pstar"); sp = j.get("specs", {}).get(ps)
        if sp is None or "theta" not in sp or "max_depth" not in sp["theta"]:
            missing.append(dict(单元=d.name, 原因="pstar 规格缺 theta.max_depth")); continue
        h = j["specs"].get("HTFS", {})
        rows.append(dict(库=db, 结局=oc, 表型=ph, seed=int(seed), 外层折=int(fold), pstar=ps, 族=sp["arch"], 权重=sp["w"], 校准=bool(sp.get("cal")),
                         P星深度=int(sp["theta"]["max_depth"]), P星min_child_weight=int(sp["theta"]["min_child_weight"]),
                         HTFS_A深度=int(h["thetaA"]["max_depth"]) if "thetaA" in h else None,
                         HTFS_B深度=int(h["thetaB"]["max_depth"]) if "thetaB" in h else None, specs_SHA16=sha16(f.read_bytes())))
    return pd.DataFrame(rows), pd.DataFrame(missing, columns=["单元", "原因"])


def c8_check(u):
    """P* 的「族|权重|raw/cal」计数 与 R42 C8（main，P* 行）逐项比较。"""
    c8 = pd.read_excel(C8, sheet_name="C8_开发选择分布")
    ref = c8[(c8.分支 == "main") & (c8.臂 == "P*")][["库", "选择", "单元数"]]
    mine = u.assign(选择=u.族 + "|" + u.权重 + "|" + u.校准.map({True: "cal", False: "raw"})).groupby(["库", "选择"]).size().rename("本次").reset_index()
    m = ref.merge(mine, on=["库", "选择"], how="outer").fillna(0)
    m["一致"] = m.单元数 == m.本次
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u, miss = read_units()
    chk = c8_check(u)
    dist = u.groupby(["库", "P星深度"]).size().unstack(fill_value=0)
    summ = u.groupby("库").agg(单元=("P星深度", "size"), 深度7=("P星深度", lambda s: int((s == 7).sum()))).reset_index()
    summ["深度7比例"] = summ.深度7 / summ.单元
    summ["数据库"] = summ.库.map(DB)
    strat = u.assign(深度7=u.P星深度 == 7).groupby(["库", "结局", "表型"]).agg(单元=("深度7", "size"), 深度7=("深度7", "sum")).reset_index()
    by_w = u.assign(深度7=u.P星深度 == 7).groupby(["库", "族", "权重"]).agg(单元=("深度7", "size"), 深度7=("深度7", "sum")).reset_index()
    htfs = pd.concat([u.groupby("库").HTFS_A深度.value_counts().rename("单元").reset_index().rename(columns={"HTFS_A深度": "深度"}).assign(轨="A"),
                      u.groupby("库").HTFS_B深度.value_counts().rename("单元").reset_index().rename(columns={"HTFS_B深度": "深度"}).assign(轨="B")])
    htfs = htfs.pivot_table(index=["库", "轨"], columns="深度", values="单元", fill_value=0)
    for n, df in (("units", u), ("P星深度分布", dist.reset_index()), ("深度7汇总", summ), ("深度7_按结局表型", strat), ("深度7_按族权重", by_w),
                  ("HTFS深度分布_附带", htfs.reset_index()), ("与R42_C8核对", chk), ("缺失或不可恢复", miss)):
        df.to_csv(OUT / f"{n}.csv", index=False, encoding="utf-8-sig")
    rec = dict(性质="一次限定范围的只读配置计数（GPT 第 1 轮 O5 授权）；未调用任何开发、拟合、预测或再校准函数；不是有效性分析",
               范围="MIMIC-IV / eICU-CRD，R42 main 分支，完整外层开发集 D", 读取单元=len(u), 缺失或不可恢复=len(miss),
               与R42_C8逐项一致=bool(chk.一致.all()), 分母核对=bool(all(strat.groupby("库").单元.sum().to_dict()[k] == int(summ.set_index("库").单元[k]) for k in summ.库)),
               深度7={DB[r.库]: f"{r.深度7}/{r.单元}" for r in summ.itertuples()},
               脚本SHA16=sha16(pathlib.Path(__file__).read_bytes()), C8_SHA16=sha16(C8.read_bytes()),
               specs文件指纹合并SHA16=sha16("".join(u.specs_SHA16).encode()))
    (OUT / "R50_DEPTH_COUNT.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(rec, ensure_ascii=False, indent=1)); print(dist); print(summ); print(by_w.to_string()); print(htfs); print(chk[~chk.一致])
    print(strat[strat.深度7 > 0].to_string())
    return rec["与R42_C8逐项一致"] and rec["分母核对"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
