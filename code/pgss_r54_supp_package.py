# -*- coding: utf-8 -*-
r"""R54 打包：只交一个 ZIP（补充材料 v0.1、长表、索引、QA、回复报告、代码与 SHA256 清单）。"""
import sys, json, hashlib, pathlib, zipfile
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = HERE.parent / "37_R54_补充材料v0.1"
DST = pathlib.Path(r"D:\Research\3_对话与交接\GPT交接_PGSS\53\4_Claude_R54补充材料v0.1（待上传）")
ZIP = DST / "PGSS_R54_Claude_Supplementary_v0.1.zip"
FILES = {"PGSS_Supplementary_v0.1.docx": OUT / "PGSS_Supplementary_v0.1.docx", "PGSS_Supplementary_v0.1_DataTables.xlsx": OUT / "PGSS_Supplementary_v0.1_DataTables.xlsx",
         "R54_SUPP_INDEX.xlsx": OUT / "R54_SUPP_INDEX.xlsx", "R54_Claude_回复.md": OUT / "R54_Claude_回复.md", "R54_QA.json": OUT / "R54_QA.json",
         "R54_PHASE1.json": OUT / "R54_PHASE1.json", "build.log": OUT / "build.log"}
CODE = ["pgss_r54_supp_common.py", "pgss_r54_supp_tables.py", "pgss_r54_supp_build.py", "pgss_r54_supp_report.py", "pgss_r54_supp_package.py"]


def main():
    q = json.loads((OUT / "R54_QA.json").read_text(encoding="utf-8"))
    assert q["QA"].split("/")[0] == q["QA"].split("/")[1], f"QA 未全过：{q['QA']}"
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
        priv = [n for n in names if n.endswith(".parquet") or "preds_test" in n or "_local_only" in n]
    others = [p.name for p in DST.iterdir() if p.name != ZIP.name]
    print("zip", ZIP.name, len(names), "files; testzip", bad, "; 逐字节相同", same, "; 逐人文件", priv, "; 文件夹内其他文件", others)
    if bad or not same or priv or others:
        raise SystemExit("打包自检未通过")


if __name__ == "__main__":
    main()
