# -*- coding: utf-8 -*-
r"""R46 交付自检的变异对照：真实 zip 与当前报告必须通过；下列每一种破坏都必须被检出。
另加**自包含检查**（GPT 第 50 轮 §1）：把 zip 解压到临时目录，用隔离解释器（python -I，不读用户 site 与当前目录）
逐个导入 code/ 下全部模块（sys.path 只含解压后的 code/ 与 code/external/）；变异对照 = 删掉 pgss_core.py 后导入引擎必须失败。
两层数字检查分别记录：① 舍入匹配溯源（selfaudit）；② 逐值追踪 + 报告 SHA（trace_audit）。每个变异只要求至少一层检出，并如实记录哪一层检出。
「改动一个两位小数百分数」预期第①层查不出（两位小数随机命中率≈1），只靠第②层。
用法：python pgss_r46_package_tests.py   （变异用的真实数字从报告中自动选取）   输出：28_R46_最终核验与T2/R46_PACKAGE_TESTS.json
"""
import sys, io, json, zipfile, pathlib, tempfile, re, subprocess, shutil, os
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r46_package as P
from pgss_r45_package_tests import mkzip, pqb


def bump(s):
    """把数字（可带 %）最后一位加 1，负数朝更负方向。"""
    m = re.fullmatch(r"(-?\d+)\.(\d+)(%?)", s)
    d = len(m.group(2)); v = float(m.group(1) + "." + m.group(2)) + (10 ** -d) * (-1 if s.startswith("-") else 1)
    return f"{v:.{d}f}{m.group(3)}"


def pick(md, pat):
    """报告中第一个匹配 pat、且只出现一次的真实数字（只出现一次 → 改动它不会被同一数字的其他出现掩盖）。"""
    for x in re.findall(pat, md):
        if md.count(x) == 1:
            return x
    raise AssertionError(f"报告中找不到可用于变异的数字：{pat}")

PY = sys.executable


def import_all(code_dir, mods):
    """隔离解释器中逐个导入；返回 {模块: 错误或 None}。"""
    src = ("import sys, importlib, json\n"
           f"sys.path[:0] = [{str(code_dir)!r}, {str(code_dir / 'external')!r}]\n"
           "res = {}\n"
           f"for m in {mods!r}:\n"
           "    try:\n"
           "        importlib.import_module(m); res[m] = None\n"
           "    except Exception as e:\n"
           "        res[m] = f'{type(e).__name__}: {e}'[:300]\n"
           "print('@@' + json.dumps(res, ensure_ascii=False))\n")
    r = subprocess.run([PY, "-I", "-c", src], cwd=str(code_dir.parent), capture_output=True, text=True, encoding="utf-8",
                       env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)} | {"PYTHONIOENCODING": "utf-8"})
    line = [l for l in r.stdout.splitlines() if l.startswith("@@")]
    return json.loads(line[0][2:]) if line else {"_解释器": r.stderr[-500:]}


def selfcontained(zb):
    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        z.extractall(tmp)
    code = tmp / "code"
    mods = sorted(p.stem for p in code.glob("*.py"))
    res = import_all(code, mods + ["nhanes"])
    ok = bool(res) and all(v is None for v in res.values())
    (code / "pgss_core.py").unlink()                     # 变异：缺依赖必须被检出
    mut = import_all(code, ["pgss_xgb_engine"])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, {k: v for k, v in res.items() if v}, len(mods), bool(mut.get("pgss_xgb_engine"))


def main():
    real, code = P.build_zip()
    tmp = pathlib.Path(tempfile.mkdtemp())
    md0 = P.MD.read_text(encoding="utf-8")
    n4 = pick(md0, r"(?<![\d.])\d+\.\d{4}(?![\d%])")
    p2 = pick(md0, r"(?<![\d.])\d+\.\d{2}%")
    R = {}
    probs, n, na = P.selfaudit(real, P.MD, P.XL); tp = P.trace_audit(P.MD)
    R["基线通过（真实 zip + 当前报告）"] = (probs == [] and tp == [], n, na["随机命中率"])
    ok, errs, nmod, mut = selfcontained(real)
    R["自包含：隔离解释器导入 code/ 全部模块"] = (ok, nmod, errs)
    R["自包含变异：删 pgss_core.py 后引擎导入失败"] = mut
    rz = io.BytesIO(real)
    cases = {
        "报告插入编造的六位小数": (real, md0 + "\n编造 0.123457\n"),
        "报告插入编造的百分数": (real, md0 + "\n编造 -0.9876%\n"),
        "报告插入编造的科学计数法": (real, md0 + "\n编造 4.2e-09\n"),
        f"报告改动一个四位小数（{n4} → {bump(n4)}）": (real, md0.replace(n4, bump(n4), 1)),
        f"报告改动一个两位小数百分数（{p2} → {bump(p2)}；预期只有第②层检出）": (real, md0.replace(p2, bump(p2), 1)),
        "zip 含逐人预测文件": (mkzip(rz, {"r46/nwicu_v2/runs/x/preds_test.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
        "zip 含本地专用目录": (mkzip(rz, {"r46/nwicu_v2/_local_only/MORT_STATUS.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0),
        "parquet 含 stay_id 列": (mkzip(rz, {"r46/x.parquet": pqb(pd.DataFrame({"stay_id": [1]}))}), md0),
        "parquet 含 subject_id 列": (mkzip(rz, {"r46/y.parquet": pqb(pd.DataFrame({"subject_id": [1]}))}), md0),
        "json 含 hadm_id 键": (mkzip(rz, {"r46/x.json": b'{"hadm_id": 3}'}), md0),
        "py 含真实口令": (mkzip(rz, {"code/b.py": ("PGPASS" + "WORD=" + "hunt" + "er2\n").encode()}), md0),
    }
    for nm, (zb, md) in cases.items():
        f = tmp / "r.md"; f.write_text(md, encoding="utf-8")
        a, _, _ = P.selfaudit(zb, f, P.XL); b = P.trace_audit(f)
        R[nm] = dict(检出=bool(a or b), 第1层_zip内容与舍入溯源检出=bool(a), 第2层_逐值追踪或报告SHA检出=bool(b))
    base = R["基线通过（真实 zip + 当前报告）"][0] and R["自包含：隔离解释器导入 code/ 全部模块"][0] and R["自包含变异：删 pgss_core.py 后引擎导入失败"]
    out = dict(全部通过=bool(base and all(v["检出"] for v in R.values() if isinstance(v, dict))), 结果=R)
    (P.S / "R46_PACKAGE_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["全部通过"] else 1)
