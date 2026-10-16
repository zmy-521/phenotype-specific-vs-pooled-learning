# -*- coding: utf-8 -*-
r"""R45 交付自检的变异对照：真实 zip 与当前报告必须通过；下列每一种破坏都必须被检出。
用法：python pgss_r45_package_tests.py <报告中一个真实存在的百分数，如 3.23%>   输出：27_R45_外部复现与机制核验/R45_PACKAGE_TESTS.json
"""
import sys, io, json, zipfile, pathlib, tempfile, re
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r45_package as P


def mkzip(base, extra):
    b = io.BytesIO()
    with zipfile.ZipFile(base) as zi, zipfile.ZipFile(b, "w") as zo:
        for n in zi.namelist():
            zo.writestr(n, zi.read(n))
        for n, d in extra.items():
            zo.writestr(n, d)
    return b.getvalue()


def pqb(df):
    b = io.BytesIO(); df.to_parquet(b, index=False); return b.getvalue()


def bump(s):
    """把百分数最后一位加 1（如 3.23% → 3.24%）。"""
    m = re.match(r"(-?\d+)\.(\d+)%", s)
    d = len(m.group(2)); v = float(s[:-1]) + (10 ** -d) * (1 if not s.startswith("-") else -1)
    return f"{v:.{d}f}%"


def main(real_pct):
    real = P.build_zip()
    tmp = pathlib.Path(tempfile.mkdtemp())
    md0 = P.MD.read_text(encoding="utf-8")
    assert real_pct in md0, "变异对照依赖的真实数字不在报告中"
    R = {}
    probs, n, na = P.selfaudit(real, P.MD, P.XL)
    R["基线通过（真实 zip + 当前报告）"] = (probs == [], n, na["随机命中率"])
    rz = io.BytesIO(real)
    cases = {
        "报告插入编造的六位小数": (real, md0 + "\n编造 0.123457\n"),
        "报告插入编造的百分数": (real, md0 + "\n编造 -0.9876%\n"),
        "报告插入编造的科学计数法": (real, md0 + "\n编造 4.2e-09\n"),
        f"报告改动一个真实数字（{real_pct} → {bump(real_pct)}）": (real, md0.replace(real_pct, bump(real_pct), 1)),
        "zip 含逐人预测文件": (mkzip(rz, {"r45/nwicu/runs/x/preds_test.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
        "zip 含本地暂存目录": (mkzip(rz, {"r45/nwicu/_local_only/stage_lab.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
        "parquet 含 stay_id 列": (mkzip(rz, {"r45/nwicu/data/X.parquet": pqb(pd.DataFrame({"stay_id": [1]}))}), md0),
        "parquet 含 subject_id 列": (mkzip(rz, {"r45/x.parquet": pqb(pd.DataFrame({"subject_id": [1]}))}), md0),
        "json 含 hadm_id 键": (mkzip(rz, {"r45/x.json": b'{"hadm_id": 3}'}), md0),
        "py 含真实口令": (mkzip(rz, {"code/b.py": ("PGPASS" + "WORD=" + "hunt" + "er2\n").encode()}), md0),  # 运行时拼出
    }
    for nm, (zb, md) in cases.items():
        f = tmp / "r.md"; f.write_text(md, encoding="utf-8")
        probs, _, _ = P.selfaudit(zb, f, P.XL)
        R[nm] = bool(probs)
    base = R["基线通过（真实 zip + 当前报告）"][0]
    out = dict(全部通过=bool(base and all(v for k, v in R.items() if not k.startswith("基线"))), 结果=R)
    (P.S / "R45_PACKAGE_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1])["全部通过"] else 1)
