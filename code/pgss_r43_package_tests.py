# -*- coding: utf-8 -*-
r"""R43 交付自检的变异对照：当前报告与无害 zip 必须通过；下列每一种破坏都必须被检出。
用法：python pgss_r43_package_tests.py   输出：24_R43_结果补齐与T1/R43_PACKAGE_TESTS.json
"""
import sys, io, json, zipfile, pathlib, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r43_package as P


def mkzip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return b.getvalue()


def pqb(df):
    b = io.BytesIO(); df.to_parquet(b, index=False); return b.getvalue()


def main():
    ok_zip = mkzip({"code/a.py": b"x = 1\n", "t1/x.parquet": pqb(pd.DataFrame({"臂": ["P*"], "R": [0.1]}))})
    tmp = pathlib.Path(tempfile.mkdtemp())
    md0 = P.MD.read_text(encoding="utf-8")
    R = {}
    probs, n, na = P.selfaudit(ok_zip, P.MD, P.XL)
    R["基线通过"] = (probs == [], n, na["随机命中率"])
    cases = {
        "报告插入编造的六位小数": (ok_zip, md0 + "\n编造 0.123457\n"),
        "报告插入编造的百分数": (ok_zip, md0 + "\n编造 -0.9876%\n"),
        "报告插入编造的科学计数法": (ok_zip, md0 + "\n编造 4.2e-09\n"),
        "zip 含逐人预测文件": (mkzip({"x/preds_test.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
        "zip 含联合重抽矩阵": (mkzip({"readonly/R43_BOOT_RISKS.parquet": pqb(pd.DataFrame({"a": [1.0]}))}), md0),
        "zip 含本地专用目录": (mkzip({"_local_only/a.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
        "parquet 含 unit_id 列": (mkzip({"t1/x.parquet": pqb(pd.DataFrame({"unit_id": [1]}))}), md0),
        "jsonl 含 subject_id 键": (mkzip({"t1/x.jsonl": b'{"subject_id": 3}\n'}), md0),
        "py 含真实口令": (mkzip({"code/b.py": ("PGPASS" + "WORD=" + "hunt" + "er2\n").encode()}), md0),  # 运行时拼出，源码不含字面值
    }
    for nm, (zb, md) in cases.items():
        f = tmp / "r.md"; f.write_text(md, encoding="utf-8")
        probs, _, _ = P.selfaudit(zb, f, P.XL)
        R[nm] = bool(probs)
    out = dict(全部通过=bool(R["基线通过"][0] and all(v for k, v in R.items() if k != "基线通过")), 结果=R)
    (P.S / "R43_PACKAGE_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
