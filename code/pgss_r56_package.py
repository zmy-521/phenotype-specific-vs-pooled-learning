# -*- coding: utf-8 -*-
r"""R56 打包：只交一个 ZIP（缺失率 xlsx、QC 报告、QC/测试 json、四个脚本与 SHA256 清单）。打包后逐字节回读自检。"""
import sys, json, hashlib, pathlib, zipfile
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = HERE.parent / "38_R56_ICU预测变量缺失率"
DST = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\53\6_Claude_R56缺失率（待上传）")
ZIP = DST / "PGSS_R56_ICU_Predictor_Missingness.zip"
FILES = {"R56_ICU_Predictor_Missingness.xlsx": OUT / "R56_ICU_Predictor_Missingness.xlsx", "R56_READONLY_QC.md": OUT / "R56_READONLY_QC.md",
         "R56_QC.json": OUT / "R56_QC.json", "R56_TESTS.json": OUT / "R56_TESTS.json"}
CODE = ["pgss_r56_missingness.py", "pgss_r56_tests.py", "pgss_r56_report.py", "pgss_r56_package.py"]


def main():
    q = json.loads((OUT / "R56_QC.json").read_text(encoding="utf-8")); t = json.loads((OUT / "R56_TESTS.json").read_text(encoding="utf-8"))
    assert q["通过"] and all(q["变异对照"].values()), "QA 未全过"
    assert t["测试"].split("/")[0] == t["测试"].split("/")[1] and t["变异对照"].split("/")[0] == t["变异对照"].split("/")[1], "测试未全过"
    DST.mkdir(parents=True, exist_ok=True)
    files = dict(FILES); files.update({f"code/{c}": HERE / c for c in CODE})
    man = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items():
            zf.write(v, k)
        zf.writestr("SHA256_MANIFEST.json", json.dumps(man, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(ZIP) as zf:
        bad = zf.testzip(); names = zf.namelist()
        same = all(zf.read(k) == pathlib.Path(v).read_bytes() for k, v in files.items())
        mani = json.loads(zf.read("SHA256_MANIFEST.json")); man_ok = all(hashlib.sha256(zf.read(k)).hexdigest() == h for k, h in mani.items())
        priv = [n for n in names if n.endswith(".parquet") or "preds_test" in n or "_local_only" in n]
    others = [p.name for p in DST.iterdir() if p.name != ZIP.name]
    print("zip", ZIP.name, len(names), "files; testzip", bad, "; 逐字节相同", same, "; 清单回验", man_ok, "; 逐人文件", priv, "; 文件夹内其他文件", others)
    print("zip sha256", hashlib.sha256(ZIP.read_bytes()).hexdigest())
    if bad or not same or not man_ok or priv or others:
        raise SystemExit("打包自检未通过")


if __name__ == "__main__":
    main()
