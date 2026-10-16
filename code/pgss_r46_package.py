# -*- coding: utf-8 -*-
r"""R46 打包与交付前自检 → 28_R46_最终核验与T2/R46_CODE_TESTS.zip。

代码：R46 全部脚本 + **按 import 关系自动求出的本地依赖闭包**（GPT 第 50 轮 §1：R45 包缺 pgss_core.py、pgss_sharing.py，不能独立导入引擎）
      + 外部辅助模块 nhanes.py（NHANES 死因核实用）。
材料：两份方案与全部冻结记录、核实与等价测试、干跑审计、折归属核对、推断（修正后与零修正比较）、只读更正、数字包、文献定位表（不含原文）、
      T2 配置/端到端测试/成本/汇总与逐重复记录（合成数据）、驱动与日志、环境记录。
**不打包**：任何 runs/ 下的逐单元产物（preds_test 含 stay_id / SEQN）、_local_only/、文献 PDF 与抽取文本、个体数据。
自检：沿用 pgss_r43_package.selfaudit（zip 内容 + 报告数字溯源），另加**自包含检查**（pgss_r46_package_tests.py 中执行：
      解压到临时目录，用隔离解释器 python -I 逐个导入 code/ 下全部模块）。
用法：python pgss_r46_package.py
"""
import sys, json, pathlib, platform, hashlib, zipfile, io, os, ast
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r43_package import selfaudit

ROOT = HERE.parent
S = ROOT / "28_R46_最终核验与T2"
MD = S / "R46_最终核验与T2_统一报告.md"; XL = S / "R46_最终核验与T2_证据表.xlsx"
NHMOD = pathlib.Path(r"D:/NAH公共数据库/NHANES/2_代码/nhanes.py")
ENTRY = ["pgss_r46_outcome_verify", "pgss_r46_rerun", "pgss_r46_inference", "pgss_r46_t2", "pgss_r46_t2_summary", "pgss_r46_t2_tests",
         "pgss_r46_readonly", "pgss_r46_numbers", "pgss_r46_lit", "pgss_r46_report", "pgss_r46_package", "pgss_r46_package_tests",
         "pgss_r46_sched_check", "pgss_r46_noeco"]
FILES = ["outcome_fix/R46_OUTCOME_FIX_PROTOCOL_v1.0.md", "outcome_fix/R46_OUTCOME_FIX_FREEZE.json", "outcome_fix/R46_RERUN_FREEZE.json",
         "outcome_fix/R46_OUTCOME_VERIFY.xlsx", "outcome_fix/R46_OUTCOME_VERIFY.json", "outcome_fix/R46_RERUN_EQUIV.json",
         "outcome_fix/R46_RERUN_PATHAUDIT.json", "outcome_fix/R46_RERUN_PATHAUDIT.csv", "outcome_fix/R46_FOLDCHECK.csv",
         "outcome_fix/R46_INFERENCE_CHECK.json", "outcome_fix/_r46_outcome_driver.py", "outcome_fix/outcome_driver.log",
         "outcome_fix/equiv.log", "outcome_fix/pathaudit.log", "outcome_fix/run_main.log", "outcome_fix/run_pop.log",
         "outcome_fix/foldcheck.log", "outcome_fix/inf_check.log",
         "nwicu_v2/inference/R45_NWICU_INFERENCE.xlsx", "nwicu_v2/inference/R45_NWICU_INFERENCE.json", "nwicu_v2/inference/_ROUTING.json",
         "nhanes_v2/r44v2/inference/R44_INFERENCE.xlsx", "nhanes_v2/r44v2/inference/R44_INFERENCE.json", "nhanes_v2/r44v2/inference/_ROUTING.json",
         "nhanes_v2/target_v2/R45_TARGET_INFERENCE.xlsx", "nhanes_v2/target_v2/R45_TARGET_INFERENCE.json", "nhanes_v2/target_v2/_ROUTING.json",
         "readonly/R46_READONLY.xlsx", "numbers/R46_NUMBERS.xlsx", "numbers/R46_NUMBERS.json",
         "lit/R46_LIT_TABLE.xlsx", "lit/R46_LIT_TABLE.json", "lit/pdf_index.json",
         "t2/R46_T2_PROTOCOL_v1.0.md", "t2/R46_T2_FREEZE.json", "t2/R46_T2_E2E_TESTS.json", "t2/R46_T2_COST.json",
         "t2/R46_T2_SUMMARY.xlsx", "t2/R46_T2_SUMMARY.json", "t2/_r46_t2_driver.py", "t2/t2_driver.log", "t2/e2e_tests.log",
         "t2/run_cost.log", "t2/run_all.log", "t2/summary.log", "t2/probe.log", "t2/probe/P2-H__r000/rep.json",
         "readonly/R46_READONLY.json", "R46_NUM_TRACE.json", "t2/R46_SCHED_CHECK.json"]


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def closure(entry=ENTRY):
    """本地模块的 import 闭包（AST 静态分析，含函数内的延迟导入）。"""
    local = {p.stem for p in HERE.glob("*.py")}
    seen, todo = set(), [e for e in entry if e in local]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        t = ast.parse((HERE / f"{m}.py").read_text(encoding="utf-8"))
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                todo += [a.name.split(".")[0] for a in n.names if a.name.split(".")[0] in local]
            elif isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in local:
                todo.append(n.module.split(".")[0])
    return sorted(f"{m}.py" for m in seen)


def env_record(code):
    import xgboost, sklearn, scipy
    return dict(python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(), numpy=np.__version__,
                pandas=pd.__version__, xgboost=xgboost.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                code_sha={f: sha((HERE / f).read_bytes()) for f in code}, nhanes_module=dict(路径=str(NHMOD), sha=sha(NHMOD.read_bytes())),
                依赖闭包=dict(入口=ENTRY, 文件数=len(code)))


def t2_records():
    """T2 逐重复记录合并（合成数据；不含 fitlog 明细，拟合次数在 rep 记录中）。"""
    reps, arms, pairs, sel = [], [], [], []
    runs = S / "t2" / "runs"
    if not runs.exists():
        return {}
    for d in sorted(runs.iterdir()):
        if (d / "_DONE").exists():
            reps.append(json.loads((d / "rep.json").read_text(encoding="utf-8")))
            for lst, fn in ((arms, "arms.parquet"), (pairs, "pairs.parquet"), (sel, "sel.parquet")):
                x = pd.read_parquet(d / fn); x.insert(0, "重复", d.name); lst.append(x)
        elif (d / "_FAILED").exists():
            reps.append(dict(tag=d.name, 状态="FAILED", 信息=(d / "_FAILED").read_text(encoding="utf-8")[:2000]))
    pq = lambda df: (lambda b: (df.to_parquet(b, index=False, compression="zstd"), b.getvalue())[1])(io.BytesIO())
    out = {"t2/runs_records/rep.jsonl": "\n".join(json.dumps(r, ensure_ascii=False, default=float) for r in reps).encode("utf-8")}
    for nm, lst in (("arms", arms), ("pairs", pairs), ("sel", sel)):
        if lst:
            out[f"t2/runs_records/{nm}.parquet"] = pq(pd.concat(lst, ignore_index=True))
    return out


def build_zip():
    code = closure()
    env = env_record(code)
    (S / "R46_ENV.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code:
            z.write(HERE / f, f"code/{f}")
        z.write(NHMOD, "code/external/nhanes.py")
        for rel in FILES:
            if (S / rel).exists():
                z.write(S / rel, f"r46/{rel}")
        for k, b in t2_records().items():
            z.writestr(f"r46/{k}", b)
        z.write(S / "R46_ENV.json", "R46_ENV.json")
        if (S / "R46_PACKAGE_TESTS.json").exists():
            z.write(S / "R46_PACKAGE_TESTS.json", "tests/R46_PACKAGE_TESTS.json")
    return buf.getvalue(), code


def trace_audit(md_path):
    """第二层：生成时逐值追踪的结果 + 报告全文 SHA（检出生成后的任何手改，包括舍入匹配查不出的两位小数）。"""
    tj = S / "R46_NUM_TRACE.json"
    if not tj.exists():
        return ["缺逐值追踪记录 R46_NUM_TRACE.json"]
    t = json.loads(tj.read_text(encoding="utf-8")); probs = []
    if t["找不到"]:
        probs.append(f"逐值追踪：{len(t['找不到'])} 个原值不在报告引用表中")
    if t["变异对照被放行"]:
        probs.append(f"逐值追踪的变异对照被放行 {t['变异对照被放行']} 个")
    if sha(pathlib.Path(md_path).read_bytes()) != t.get("报告sha"):
        probs.append("报告与逐值追踪登记的 SHA 不符（生成后被改动）")
    return probs


def audit_all(zb, md_path, xl_path):
    probs, nn, na = selfaudit(zb, md_path, xl_path)
    return probs + trace_audit(md_path), nn, na


def main():
    missing = [rel for rel in FILES if not (S / rel).exists()]
    zb, code = build_zip()
    probs, nn, na = audit_all(zb, MD, XL)
    if missing:
        probs.append(f"应交付文件缺失：{missing}")
    print(f"zip {len(zb)/1024/1024:.2f} MB，sha256 {sha(zb)}；代码 {len(code)} 个；报告数字 {nn} 个；随机命中率 {na['随机命中率']}")
    if probs:
        print("自检未通过：\n  " + "\n  ".join(probs)); return False
    (S / "R46_CODE_TESTS.zip").write_bytes(zb)
    (S / "R46_PACKAGE_AUDIT.json").write_text(json.dumps(dict(zip_sha=sha(zb), zip_mb=round(len(zb) / 2**20, 2), 代码文件=len(code),
                                                              report_numbers=nn, numaudit=na, problems=[]),
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("自检通过，已写 R46_CODE_TESTS.zip")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
