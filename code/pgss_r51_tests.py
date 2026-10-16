# -*- coding: utf-8 -*-
r"""R51 包测试：每条检查配一个必须失败的变异对照。
1 md 中每个数字都已登记（来自结果表或定义常量）；2 报告源码不含手填数字；3 md 由当前 xlsx 确定性重生成；
4 md 里的 Table 1 与 xlsx T0 逐格相同；5 json 与 xlsx 质控计数一致；6 json 记录的脚本 SHA = 当前基线脚本；
7 交付工作簿只含汇总；8 交付副本与工作目录逐字节相同（打包后运行）。
"""
import sys, json, re, hashlib, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import pgss_r51_report as RP

OUT = HERE.parent / "34_R51_基线只读提数"
YEARS = {str(y) for y in range(1999, 2009)} | {"2021"}
STRIP = [r"`[^`]*`", r"(?m)^#+\s*\d+\.\s*", r"(?m)^\s*\d+\.\s", r"第\s*\d+\s*(?:节|步|条|层)", r"§\d+", r"S1-[a-h]", r"E-\d"]


def tokens(md):
    for pat in STRIP:
        md = re.sub(pat, " ", md)
    return [RP.norm(t) for t in RP.TOKEN_RE.findall(md)]


def unregistered(md, trace):
    return sorted({t for t in tokens(md) if t not in trace and t not in YEARS})


def lint(src):
    """n()/p() 的第一个参数不得是数字字面量；reg() 的数字字面量只允许登记为定义常量。"""
    bad = re.findall(r"\b[np]\(\s*[−-]?\d[\d_.]*\s*,", src)
    bad += [m for m in re.findall(r"reg\(\s*['\"][\d.,%]+['\"]\s*,\s*([^)]+)\)", src) if m.strip() != "K"]
    return bad


def sha16f(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def run(deliver=None):
    res = []
    def add(name, ok, mut, det=""):
        res.append(dict(检查=name, 通过=bool(ok), 变异对照检出=bool(mut), 说明=det))
    md = (OUT / "R51_Baseline_QC.md").read_text(encoding="utf-8")
    trace = json.loads((OUT / "R51_NUM_TRACE.json").read_text(encoding="utf-8"))
    u = unregistered(md, trace)
    add("1 md 中全部数字已登记", not u, bool(unregistered(md + "\n虚构 12,345 人，占 7.77%。", trace)), f"{len(set(tokens(md)))} 个不同数字；未登记 {u[:10]}")
    src = (HERE / "pgss_r51_report.py").read_text(encoding="utf-8")
    add("2 报告源码不含手填数字", not lint(src), bool(lint(src + "\nx = n(222, 'T5')\n")), f"{len(lint(src))} 处")
    md2 = RP.main(write=False)
    add("3 md 由当前 xlsx 确定性重生成（逐字相同）", md2 == md, md2 != md.replace("必改", "必须改", 1))
    W = pd.read_excel(OUT / "R51_Table1_Baseline_Source.xlsx", sheet_name=None)
    T0 = W["T0_Table1_排版"]
    cells = [str(v).replace("|", "/") for v in T0.drop(columns=["区块"]).to_numpy().ravel()]
    miss = [c for c in cells if f"| {c} |" not in md and f"| {c}" not in md]
    T0m = list(cells); T0m[10] = T0m[10] + "x"
    add("4 md 的 Table 1 与 xlsx T0 逐格相同", not miss, bool([c for c in T0m if f"| {c} |" not in md and f"| {c}" not in md]), f"{len(cells)} 格")
    J = json.loads((OUT / "R51_BASELINE.json").read_text(encoding="utf-8")); Q = W["A2_质控"]
    qs = f"{int(Q.通过.sum())}/{len(Q)}"
    add("5 json 质控计数 = xlsx A2", J["质控"] == qs and J["变异对照"] == f"{int(Q.变异对照检出.sum())}/{len(Q)}", J["质控"] != qs.replace("/", "//"))
    cur = sha16f(HERE / "pgss_r51_baseline.py")
    add("6 json 的脚本 SHA = 当前基线脚本（产物来自当前版本）", J["脚本SHA16"] == cur, J["脚本SHA16"] != cur[::-1])
    forbid = {"SEQN", "subject_id", "hadm_id", "stay_id", "patientunitstayid", "uniquepid", "group_id", "uid", "sid"}
    small = lambda ws: all(len(df) <= 300 and not (set(map(str, df.columns)) & forbid) for df in ws.values())
    Wm = dict(W); Wm["x"] = pd.DataFrame({"stay_id": [1]})
    add("7 交付工作簿只含汇总", small(W), not small(Wm), f"{len(W)} 张表")
    if deliver:
        pairs = [(OUT / f, pathlib.Path(deliver) / f) for f in ("R51_Baseline_QC.md", "R51_Table1_Baseline_Source.xlsx")]
        same = all(a.read_bytes() == b.read_bytes() for a, b in pairs)
        add("8 交付副本与工作目录逐字节相同", same, pairs[0][0].read_bytes() != pairs[0][1].read_bytes() + b" ", deliver)
    df = pd.DataFrame(res)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(df.to_string()); print(f"通过 {int(df.通过.sum())}/{len(df)}；变异对照检出 {int(df.变异对照检出.sum())}/{len(df)}")
    return df


if __name__ == "__main__":
    df = run(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if df.通过.all() and df.变异对照检出.all() else 1)
