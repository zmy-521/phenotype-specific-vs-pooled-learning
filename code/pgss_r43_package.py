# -*- coding: utf-8 -*-
r"""R43 打包与交付前自检 → 24_R43_结果补齐与T1/R43_CODE_TESTS.zip。

内容：代码（R43 脚本 + 复用的冻结依赖）、环境记录、只读补齐（xlsx/json/变异对照/本地复算结果/GPT 复核脚本本机运行结果）、
      T1 与 T1 扩展（配置、端到端检查、成本、汇总、逐重复记录合并；合成数据）、图、日志。
**不打包**（第48轮补充裁决）：_local_only/（簇级损失）、R42 的逐人预测 preds_*、联合重抽矩阵 R43_BOOT_RISKS.parquet、
任何含 unit_id 的文件。自检对这些名称直接判失败。

自检（任一失败即不写 zip）
  - 无 _local_only、无 preds_*、无 _withheld；parquet 无标识列；json/jsonl 无标识键；无真实口令；
  - 报告数字溯源：md 中 1–8 位小数的数字须能在证据表中按同位数四舍五入找到（原值或 ×100）；
    科学计数法数字须能在证据表中按同一格式找到。（R42 的自检只覆盖 1–4 位小数，本版补齐。）
用法：python pgss_r43_package.py
"""
import sys, json, pathlib, platform, hashlib, zipfile, re, io, os
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
S = ROOT / "24_R43_结果补齐与T1"
CODE_FILES = ["pgss_r43_readonly.py", "pgss_r43_readonly_tests.py", "pgss_r43_t1.py", "pgss_r43_t1_tests.py",
              "pgss_r43_t1_ext.py", "pgss_r43_t1_ext_tests.py", "pgss_r43_boot_verify.py", "pgss_r43_boot_verify_tests.py",
              "pgss_r43_numaudit.py", "pgss_r43_package_tests.py",
              "pgss_r43_t1_summary.py", "pgss_r43_figures.py", "pgss_r43_report.py", "pgss_r43_package.py",
              # 复用的冻结依赖（只读）
              "pgss_xgb_engine.py", "pgss_xgb_run.py", "pgss_xgb_pathaudit.py", "pgss_xgb_sim0.py", "pgss_r42_inference.py",
              "pgss_r36_engine.py", "pgss_r38_bench.py", "pgss_r381_joint_inference.py", "pgss_dataver.py",
              "pgss_threads.py", "pgss_core.py", "pgss_sharing.py"]
ID_COLS = {"unit_id", "group_id", "hospital_id", "subject_id", "stay_id", "hadm_id", "patientunitstayid",
           "uniquepid", "patienthealthsystemstayid", "uid"}
MD = S / "R43_结果补齐与T1_统一报告.md"; XL = S / "R43_结果补齐与T1_证据表.xlsx"


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def env_record():
    import xgboost, sklearn, scipy
    import pgss_dataver as DV, pgss_r38_bench as BN
    return dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(), numpy=np.__version__,
                pandas=pd.__version__, xgboost=xgboost.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                data_version=DV.ACTIVE, data_loaded_fingerprint=DV.loaded_fingerprint_all(BN.load),
                code_sha={f: sha((HERE / f).read_bytes()) for f in CODE_FILES},
                driver_sha={f: sha((S / f).read_bytes()) for f in ("_r43_t1_driver.py", "_r43_t1_ext_driver.py", "_r43_post.py")})


def pq(df):
    b = io.BytesIO(); df.to_parquet(b, index=False, compression="zstd"); return b.getvalue()


def t1_records(runs="runs", tag="runs_records"):
    reps, arms, sel, fl = [], [], [], []
    for d in sorted((S / "t1" / runs).iterdir()):
        st = "DONE" if (d / "_DONE").exists() else ("FAILED" if (d / "_FAILED").exists() else "INCOMPLETE")
        if st == "DONE":
            r = json.loads((d / "rep.json").read_text(encoding="utf-8"))
            if (d / "ext.json").exists():
                r["ext"] = json.loads((d / "ext.json").read_text(encoding="utf-8"))
            reps.append(r)
            for lst, fn in ((arms, "arms.parquet"), (sel, "sel.parquet"), (fl, "fitlog.parquet")):
                x = pd.read_parquet(d / fn); x.insert(0, "重复", d.name); lst.append(x)
        else:
            reps.append(dict(tag=d.name, 状态=st, 信息=((d / "_FAILED").read_text(encoding="utf-8")[:2000] if st == "FAILED" else "")))
    out = {f"t1/{tag}/rep.jsonl": "\n".join(json.dumps(r, ensure_ascii=False, default=float) for r in reps).encode("utf-8")}
    if arms:
        out[f"t1/{tag}/arms.parquet"] = pq(pd.concat(arms, ignore_index=True))
        out[f"t1/{tag}/sel.parquet"] = pq(pd.concat(sel, ignore_index=True))
        out[f"t1/{tag}/fitlog.parquet"] = pq(pd.concat(fl, ignore_index=True))
    return out, len(reps)


def selfaudit(zbytes, md_path, xlsx_path):
    """zip 内容检查 + 报告数字溯源（数值池 = 证据表「目录」中声明为报告引用的汇总表）。返回 (问题列表, 数字个数, 溯源详情)。"""
    import pgss_r43_numaudit as NA
    probs = []
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for n in z.namelist():
            base = pathlib.PurePosixPath(n).name
            if "_withheld" in n or "_local_only" in n or base.startswith("preds_") or "BOOT_RISKS" in base:
                probs.append(f"含逐人预测、隔离或本地专用文件：{n}")
            data = z.read(n)
            if n.endswith((".py", ".md", ".json", ".jsonl", ".txt", ".csv", ".log")):
                t = data.decode("utf-8", errors="ignore")
                for m in re.finditer(r"(?i)(password|PGPASSWORD)\s*[=:]\s*['\"]?([^\s'\"]+)", t):
                    if m.group(2).strip("<>").lower() not in ("...", "xxx", "<password>", "os.environ", "占位符", "''", '""'):
                        if not m.group(2).startswith(("os.", "$", "<", "{")):
                            probs.append(f"疑似口令 {n}: {m.group(0)[:40]}")
                if n.endswith((".json", ".jsonl")):
                    for k in ID_COLS:
                        if re.search(rf'"{k}"\s*:', t):
                            probs.append(f"含标识键 {k}：{n}")
            if n.endswith(".parquet"):
                cols = pd.read_parquet(io.BytesIO(data)).columns
                if set(cols) & ID_COLS:
                    probs.append(f"含标识列 {n}: {sorted(set(cols) & ID_COLS)}")
    xl = pd.read_excel(xlsx_path, sheet_name=None)
    cited = xl["目录"].loc[xl["目录"]["报告引用"] == "是", "表"].tolist()
    res = NA.audit(md_path.read_text(encoding="utf-8"), xl, cited)
    if res["找不到"]:
        probs.append(f"报告中 {len(res['找不到'])} 个数字在引用的汇总表中找不到：{res['找不到'][:15]}")
    return probs, res["带小数数字"] + res["科学计数法数字"], res


def main():
    env = env_record()
    (S / "R43_ENV.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    recs, n_reps = t1_records()
    if (S / "t1" / "runs_ext").exists():
        rx, n_x = t1_records("runs_ext", "runs_records_ext"); recs.update(rx); n_reps += n_x
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE_FILES:
            z.write(HERE / f, f"code/{f}")
        for f in ("_r43_t1_driver.py", "_r43_t1_ext_driver.py", "_r43_post.py"):
            z.write(S / f, f"code/{f}")
        z.write(ROOT / "00_PROTOCOL" / "PGSS_XGB_SAP_v1.0.md", "PGSS_XGB_SAP_v1.0.md")
        z.write(S / "R43_ENV.json", "R43_ENV.json")
        for fn in ("R43_READONLY.xlsx", "R43_READONLY.json", "R43_READONLY_TESTS.json", "R43_BOOT_VERIFY.json",
                   "R43_BOOT_VERIFY_TESTS.json", "R42_NUMAUDIT.json", "GPT_review_record_checks_本机运行.json"):
            z.write(S / "readonly" / fn, f"readonly/{fn}")
        for fn in ("R43_T1_CONFIG.json", "R43_T1_E2E_TESTS.json", "_COST.json", "R43_T1_SUMMARY.xlsx", "R43_T1_SUMMARY.json",
                   "R43_T1_EXT_CONFIG.json", "R43_T1_EXT_E2E_TESTS.json", "_EXT_COST.json", "R43_T1_EXT_SUMMARY.xlsx",
                   "R43_T1_EXT_SUMMARY.json"):
            if (S / "t1" / fn).exists():
                z.write(S / "t1" / fn, f"t1/{fn}")
        for k, b in recs.items():
            z.writestr(k, b)
        R44D = ROOT / "25_R44_NHANES跨领域复现"            # R44 方案先于结果交付（只含定义、指纹与审计，不含任何拟合结果）
        for rel in ("R44_NHANES_PROTOCOL_v1.0.md", "R44_FREEZE.json", "R44_PATHAUDIT.json", "data/COHORT.json"):
            z.write(R44D / rel, f"r44_nhanes_protocol/{rel}")
        for f in ("pgss_r44_nhanes_cohort.py", "pgss_r44_nhanes_run.py", "pgss_r44_nhanes_inference.py", "pgss_r43_t1_points.py"):
            z.write(HERE / f, f"code/{f}")
        z.write(R44D / "_r44_driver.py", "code/_r44_driver.py")
        NWD = ROOT / "26_R45_NWICU外部数据准备"                # NWICU 覆盖审计（只含汇总计数）
        for fn in ("R45_NWICU_AUDIT.xlsx", "R45_NWICU_AUDIT.json"):
            z.write(NWD / fn, f"r45_nwicu_audit/{fn}")
        z.write(HERE / "pgss_r45_nwicu_audit.py", "code/pgss_r45_nwicu_audit.py")
        for png in sorted((S / "figures").glob("*.png")):
            z.write(png, f"figures/{png.name}")
        if (S / "R43_PACKAGE_TESTS.json").exists():
            z.write(S / "R43_PACKAGE_TESTS.json", "tests/R43_PACKAGE_TESTS.json")
        for fn in ("readonly_run.log", "t1_driver.log", "t1_ext_driver.log"):
            if (S / fn).exists():
                z.write(S / fn, f"logs/{fn}")
        for st in sorted((S / "t1").glob("_STATUS_*.json")):
            z.write(st, f"logs/{st.name}")
    zb = buf.getvalue()
    probs, nn, na = selfaudit(zb, MD, XL)
    print(f"zip {len(zb)/1024/1024:.1f} MB，sha256 {sha(zb)}；T1 重复记录 {n_reps}；报告数字 {nn} 个")
    if probs:
        print("自检未通过：\n  " + "\n  ".join(probs)); return False
    (S / "R43_CODE_TESTS.zip").write_bytes(zb)
    (S / "R43_PACKAGE_AUDIT.json").write_text(json.dumps(dict(zip_sha=sha(zb), zip_mb=round(len(zb) / 2**20, 2), t1_reps=n_reps,
                                                              report_numbers=nn, numaudit=na, problems=[]),
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("自检通过，已写 R43_CODE_TESTS.zip")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
