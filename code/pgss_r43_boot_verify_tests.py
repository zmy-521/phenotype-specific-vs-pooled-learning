# -*- coding: utf-8 -*-
r"""本地复算检查的变异对照：把只读补齐表中一个区间下界改动 1e-6、把一个三态判定改掉、把重抽矩阵某列整体加 1e-6，
复算必须报告不一致。用法：python pgss_r43_boot_verify_tests.py；输出 readonly/R43_BOOT_VERIFY_TESTS.json
"""
import sys, json, pathlib, shutil, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r43_boot_verify as V

SRC = V.RO


def run_in(tmp):
    V.RO = tmp
    try:
        V.main()
        return json.loads((tmp / "R43_BOOT_VERIFY.json").read_text(encoding="utf-8"))
    finally:
        V.RO = SRC


def fresh():
    t = pathlib.Path(tempfile.mkdtemp())
    for f in ("R43_BOOT_RISKS.parquet", "R43_READONLY.xlsx"):
        shutil.copy(SRC / f, t / f)
    return t


def edit_xlsx(t, sheet, fn):
    X = pd.read_excel(t / "R43_READONLY.xlsx", sheet_name=None); X[sheet] = fn(X[sheet])
    with pd.ExcelWriter(t / "R43_READONLY.xlsx", engine="openpyxl") as xw:
        for k, v in X.items():
            v.to_excel(xw, sheet_name=k, index=False)


def main():
    R = {}
    t = fresh(); R["基线全部一致"] = run_in(t)["全部一致"]

    def bump(df):
        df.loc[0, "相对_P2_5"] += 1e-6; return df
    t = fresh(); edit_xlsx(t, "5_Brier对照含新增", bump); r = run_in(t)
    R["变异_区间下界改动被检出"] = r["结果"]["brier"]["相对下界最大差"] > 1e-7

    def flip(df):
        df.loc[0, "三态_m0.010"] = "支持实质收益" if df.loc[0, "三态_m0.010"] != "支持实质收益" else "排除实质收益"; return df
    t = fresh(); edit_xlsx(t, "5_Brier对照含新增", flip); R["变异_三态改动被检出"] = run_in(t)["结果"]["brier"]["三态不一致"] > 0
    t = fresh(); B = pd.read_parquet(t / "R43_BOOT_RISKS.parquet")
    c = [x for x in B.columns if x.endswith("|brier|HTFS|A")][0]; B[c] = B[c] + 1e-6
    B.to_parquet(t / "R43_BOOT_RISKS.parquet", index=False); r = run_in(t)
    R["变异_矩阵改动被检出"] = (not r["全部一致"]) or r["结果"]["brier"]["相对上界最大差"] > 1e-9
    out = dict(全部通过=all(R.values()), 结果=R)
    (SRC / "R43_BOOT_VERIFY_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
