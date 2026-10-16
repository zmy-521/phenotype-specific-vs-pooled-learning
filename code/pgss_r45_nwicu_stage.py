# -*- coding: utf-8 -*-
r"""R45 NWICU 首次构建·第 1 步：从原始 CSV 只读抽取所需记录到本地暂存（E 盘原始数据原地读取，不搬动、不修改）。

抽取范围（itemid 映射见 pgss_r45_nwicu_build.MAP；全部来自 d_items / d_labitems 的逐项核对）
  labevents ：16 项化验（含白蛋白，只用于表型）的全部记录（不在此处做时间窗，窗口在构建步骤按 stay 施加）
  chartevents：心率、动脉/无创收缩压与舒张压、呼吸频率、SpO2、身高（英寸）、体重（盎司）、R BMI（仅描述，不进构建）
另输出逐 itemid 的单位分布与分位数（单位核对表），供构建脚本的单位断言使用。
输出（**只留本地**，含 subject_id/hadm_id/stay_id）：27_R45_外部复现与机制核验/nwicu/_local_only/stage_{lab,chart}.parquet
汇总（无标识）：27_R45_外部复现与机制核验/nwicu/R45_NWICU_UNITS.xlsx
"""
import sys, json, pathlib, hashlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

SRC = pathlib.Path(r"E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data")
OUT = HERE.parent / "27_R45_外部复现与机制核验" / "nwicu"; LOC = OUT / "_local_only"
LAB_IDS = [100001, 100045, 100002, 100003, 100004, 100006, 100007, 100008, 100010, 100011, 100012, 100013, 100014,
           100016, 100017, 100021, 100040, 100041]
CHART_IDS = [320045, 320050, 320051, 320179, 320180, 320210, 320277, 326707, 326531, 300001]


def sha16_file(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time(); LOC.mkdir(parents=True, exist_ok=True)
    lab = pd.concat([c[c.itemid.isin(LAB_IDS)] for c in pd.read_csv(
        SRC / "nw_hosp/labevents.csv.gz", chunksize=2_000_000,
        usecols=["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom"],
        dtype={"value": str, "valueuom": str})], ignore_index=True)
    lab["charttime"] = pd.to_datetime(lab.charttime); lab["storetime"] = pd.to_datetime(lab.storetime)
    print(f"lab {len(lab):,} 行 {time.time()-t0:.0f}s", flush=True)
    ch = pd.concat([c[c.itemid.isin(CHART_IDS)] for c in pd.read_csv(
        SRC / "nw_icu/chartevents.csv.gz", chunksize=2_000_000,
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom"],
        dtype={"value": str, "valueuom": str})], ignore_index=True)
    ch["charttime"] = pd.to_datetime(ch.charttime); ch["storetime"] = pd.to_datetime(ch.storetime)
    print(f"chart {len(ch):,} 行 {time.time()-t0:.0f}s", flush=True)
    lab.to_parquet(LOC / "stage_lab.parquet", index=False); ch.to_parquet(LOC / "stage_chart.parquet", index=False)
    rows = []
    for nm, d in (("labevents", lab), ("chartevents", ch)):
        for (it, u), h in d.groupby(["itemid", d.valueuom.fillna("<缺失>")]):
            v = h.valuenum
            rows.append(dict(表=nm, itemid=int(it), 单位=u, 记录=len(h), valuenum缺失=int(v.isna().sum()),
                             value非数值=int((v.isna() & h.value.notna()).sum()),
                             charttime缺失=int(h.charttime.isna().sum()), storetime缺失=int(h.storetime.isna().sum()),
                             storetime早于charttime=int((h.storetime < h.charttime).sum()),
                             P0_1=v.quantile(.001), P1=v.quantile(.01), P50=v.median(), P99=v.quantile(.99), P99_9=v.quantile(.999)))
    U = pd.DataFrame(rows)
    src = {f: sha16_file(SRC / f) for f in ("nw_hosp/labevents.csv.gz", "nw_icu/chartevents.csv.gz", "nw_hosp/admissions.csv.gz",
                                              "nw_hosp/patients.csv.gz", "nw_icu/icustays.csv.gz", "nw_icu/d_items.csv.gz",
                                              "nw_hosp/d_labitems.csv.gz")}
    official = {}
    for line in (SRC.parent / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            official[parts[1].replace("data/", "")] = parts[0]
    chk = [dict(文件=f, 本地SHA256=s, 官方清单=official.get(f), 一致=(official.get(f) == s)) for f, s in src.items()]
    with pd.ExcelWriter(OUT / "R45_NWICU_UNITS.xlsx", engine="openpyxl") as xw:
        U.to_excel(xw, sheet_name="1_单位与分位数", index=False)
        pd.DataFrame(chk).to_excel(xw, sheet_name="2_源文件SHA256", index=False)
    (OUT / "R45_NWICU_STAGE.json").write_text(json.dumps(dict(
        lab行=len(lab), chart行=len(ch), 源文件=chk, 秒=round(time.time() - t0),
        暂存SHA={p.name: sha16_file(p)[:16] for p in sorted(LOC.glob("stage_*.parquet"))}),
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(U.to_string()); print(pd.DataFrame(chk).to_string())


if __name__ == "__main__":
    main()
