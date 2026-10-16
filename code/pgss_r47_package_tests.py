# -*- coding: utf-8 -*-
r"""R47 包测试：每项检查配一个必须被检出的变异对照。结果写 29_R47_写作交接/R47_PACKAGE_TESTS.json（报告表 A4 读取）。

  1 自包含：zip 解压到临时目录，隔离解释器（python -I）逐个导入 code/ 全部模块；变异 = 删 pgss_core.py 后引擎导入必须失败。
  2 合成示例 quick：在解压目录中运行（缩小网格），退出码 0 且写出结果。
  3 合成示例 reproduce：在解压目录中用冻结代码重新运行 T2 的 P2 第 0 次重复，与解压出的 t2_records 逐位相同；
    变异 = 把冻结记录中一个浮点值乘 1 + 1e-12 后，同一比较函数必须判为不同。
  4 图可由图源重现：用解压出的代码从交付的数字包重画六张图，PNG 与包内逐字节相同；变异 = 改动 F5 图源一个值后重画 Fig5，PNG 必须不同。
  5 数字溯源：真实 zip 与当前报告通过两层检查；编造或改动数字、加入患者级文件或口令时至少一层检出。
用法：python pgss_r47_package_tests.py
"""
import sys, io, json, zipfile, pathlib, tempfile, re, subprocess, shutil, os, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r47_package as P
from pgss_r46_package_tests import bump, pick
from pgss_r45_package_tests import mkzip, pqb

PY = sys.executable
ENV = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)} | {"PYTHONIOENCODING": "utf-8"}
TESTS = P.S / "tests"
PGVAR = "PG" + "PASSWORD"


def run(args, cwd, timeout=1800):
    t0 = time.time()
    r = subprocess.run([PY, "-I", "-X", "utf8"] + args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV,
                       timeout=timeout)
    return r.returncode, r.stdout[-3000:], r.stderr[-2000:], round(time.time() - t0, 1)


def import_all(code_dir, mods, env_extra=None):
    """隔离解释器（-I，-X utf8）逐个导入；SystemExit 也捕获（数据构建脚本在缺数据库口令环境变量时按设计在导入时退出）。"""
    src = ("import sys, importlib, json\n"
           f"sys.path[:0] = [{str(code_dir)!r}, {str(code_dir / 'external')!r}]\n"
           "res = {}\n"
           f"for m in {mods!r}:\n"
           "    try:\n"
           "        importlib.import_module(m); res[m] = None\n"
           "    except BaseException as e:\n"
           "        res[m] = f'{type(e).__name__}: {e}'[:300]\n"
           "print('@@' + json.dumps(res, ensure_ascii=False))\n")
    r = subprocess.run([PY, "-I", "-X", "utf8", "-c", src], cwd=str(code_dir.parent), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV | (env_extra or {}))
    line = [l for l in r.stdout.splitlines() if l.startswith("@@")]
    return json.loads(line[0][2:]) if line else {"_解释器": (r.stderr or "")[-500:]}


def main():
    rows = []
    add = lambda 检查, 通过, 说明="": rows.append(dict(检查=检查, 通过=bool(通过), 说明=说明))
    TESTS.mkdir(parents=True, exist_ok=True)
    zb, code = P.build_zip(with_tests=False)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="r47pkg_"))
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        z.extractall(tmp)
    cd = tmp / "code"
    # 1 自包含
    mods = sorted(p.stem for p in cd.glob("*.py"))
    res = import_all(cd, mods + ["nhanes"], {PGVAR: "placeholder-not-a-password"})      # 只为让数据构建脚本走完导入；不连接数据库
    bad = {k: v for k, v in res.items() if v}
    add("自包含：隔离解释器导入 code/ 全部模块", res and not bad, f"{len(mods)} 个模块 + external/nhanes" + (f"；失败 {bad}" if bad else ""))
    noenv = import_all(cd, ["pgss_r383_build_v5"])
    add("数据构建脚本在没有口令环境变量时拒绝运行（导入即退出）", str(noenv.get("pgss_r383_build_v5", "")).startswith("SystemExit"),
        str(noenv.get("pgss_r383_build_v5"))[:80])
    # 2 quick
    rc, out, err, sec = run([str(cd / "pgss_r47_example.py"), "quick", "--out", str(tmp / "ex_quick")], tmp)
    jq = tmp / "ex_quick" / "EXAMPLE_RESULT.json"
    add("合成示例 quick（解压目录中运行，缩小网格）", rc == 0 and jq.exists(), f"退出码 {rc}，{sec} 秒" + ("" if rc == 0 else f"；{err[-300:]}"))
    if jq.exists():
        shutil.copy(jq, TESTS / "example_quick.json")
    # 3 reproduce
    rc, out, err, sec = run([str(cd / "pgss_r47_example.py"), "reproduce", "--out", str(tmp / "ex_repro"), "--records", str(tmp / "t2_records")], tmp,
                            timeout=3600)
    jr = tmp / "ex_repro" / "EXAMPLE_RESULT.json"
    rr = json.loads(jr.read_text(encoding="utf-8")) if jr.exists() else {}
    add("合成示例 reproduce：重新运行冻结 T2 重复 P2 #0，与交付记录逐位相同", rc == 0 and rr.get("逐位相同") is True,
        f"退出码 {rc}，{sec} 秒；" + "；".join(f"{k}：{'相同' if v.get('相同') else '不同'}" for k, v in rr.get("结果", {}).items()))
    if jr.exists():
        shutil.copy(jr, TESTS / "example_reproduce.json")
        sys.path.insert(0, str(cd))
        import importlib
        EX = importlib.import_module("pgss_r47_example")
        fz = EX.load_frozen(tmp / "t2_records", rr["重复"]); new = pd.read_parquet(tmp / "ex_repro" / rr["重复"] / "sel.parquet")
        m = fz["sel"].copy(); c = "R"
        i = int(m.index[m[c].notna() & (m[c] != 0)][0]); m.loc[i, c] = m.loc[i, c] * (1 + 1e-12)      # 取有限非零值（NaN 乘任何数仍是 NaN）
        add("reproduce 变异对照：冻结记录一个浮点值乘 1 + 1e-12 必须被判为不同", not EX.compare_df(m, new)["相同"], f"sel 列 {c} 第 {i} 行")
    # 4 图
    fo = tmp / "fig_re"
    rc, out, err, sec = run([str(cd / "pgss_r47_figures.py"), str(P.XL), str(fo)], tmp)
    same = [p.name for p in sorted((tmp / "figures").glob("*.png")) if (fo / p.name).exists() and (fo / p.name).read_bytes() == p.read_bytes()]
    add("图可由数字包图源重现：六张 PNG 逐字节相同", rc == 0 and len(same) == 6, f"{len(same)}/6；{sec} 秒")
    xl2 = tmp / "mut.xlsx"
    with pd.ExcelWriter(xl2, engine="openpyxl") as xw:
        for k, v in pd.read_excel(P.XL, sheet_name=None).items():
            if k == "F5_T2机制":
                v = v.copy(); v.loc[0, "真值相对收益均值"] = v.loc[0, "真值相对收益均值"] + 0.005
            v.to_excel(xw, sheet_name=k, index=False)
    fm = tmp / "fig_mut"
    src = (f"import sys; sys.argv=['x', {str(xl2)!r}, {str(fm)!r}]; sys.path.insert(0, {str(cd)!r})\n"
           "import pgss_r47_figures as F; F.fig5()")
    r = subprocess.run([PY, "-I", "-c", src], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    p5 = "R47_Fig5_T2_mechanisms.png"
    add("图变异对照：改动 F5 图源一个值后 Fig5 必须不同", r.returncode == 0 and (fm / p5).exists() and (fm / p5).read_bytes() != (tmp / "figures" / p5).read_bytes())
    # 5 数字溯源
    md0 = P.MD.read_text(encoding="utf-8")
    probs, n, na = P.audit_all(zb, P.MD, P.XL)
    add("数字溯源基线：真实 zip 与当前报告两层均通过", not probs, f"报告数字 {n} 个；随机命中率 {na['随机命中率']}" + (f"；{probs}" if probs else ""))
    try:
        n4 = pick(md0, r"(?<![\d.])\d+\.\d{6}(?![\d%])")
    except AssertionError:
        n4 = None
    p2 = pick(md0, r"(?<![\d.])\d+\.\d{2}%")
    rz = io.BytesIO(zb)
    # 检查名不含具体数字（报告会列出检查名；数字只放在说明里，报告不打印说明）
    cases = {"报告插入编造的六位小数": (zb, md0 + "\n编造 0.123457\n", ""),
             "报告插入编造的百分数": (zb, md0 + "\n编造 -0.9876%\n", ""),
             "报告改动一个两位小数百分数": (zb, md0.replace(p2, bump(p2), 1), f"{p2} → {bump(p2)}"),
             "zip 含逐人预测文件": (mkzip(rz, {"x/runs/u/preds_test.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0, ""),
             "zip 含本地专用目录": (mkzip(rz, {"x/_local_only/a.parquet": pqb(pd.DataFrame({"a": [1]}))}), md0, ""),
             "parquet 含 stay_id 列": (mkzip(rz, {"x/y.parquet": pqb(pd.DataFrame({"stay_id": [1]}))}), md0, ""),
             "json 含 subject_id 键": (mkzip(rz, {"x/y.json": b'{"subject_id": 3}'}), md0, ""),
             "py 含真实口令": (mkzip(rz, {"code/b.py": ("PGPASS" + "WORD=" + "hunt" + "er2\n").encode()}), md0, "")}
    if n4:
        cases["报告改动一个六位小数"] = (zb, md0.replace(n4, bump(n4), 1), f"{n4} → {bump(n4)}")
    for nm, (z_, md, det) in cases.items():
        f = tmp / "r.md"; f.write_text(md, encoding="utf-8")
        a, _, _ = P.selfaudit(z_, f, P.XL); b = P.trace_audit(f)
        add(f"变异：{nm}", bool(a or b), (f"{det}；" if det else "") + f"第①层{'检出' if a else '未检出'}；第②层{'检出' if b else '未检出'}")
    # 1 的变异放最后（删文件）
    (cd / "pgss_core.py").unlink()
    mut = import_all(cd, ["pgss_xgb_engine"])
    add("自包含变异：删 pgss_core.py 后引擎导入必须失败", bool(mut.get("pgss_xgb_engine")))
    shutil.rmtree(tmp, ignore_errors=True)
    out = dict(全部通过=all(r["通过"] for r in rows), 检查=rows, 代码文件=len(code))
    (P.S / "R47_PACKAGE_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["全部通过"] else 1)
