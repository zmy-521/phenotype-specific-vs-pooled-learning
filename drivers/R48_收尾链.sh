#!/bin/sh
# R48 收尾链：报告（写 xlsx）→ 图 → 包测试 → 报告（含 A4）→ 打包自检
cd "D:/Research/2_课题/PGSS_表型引导选择性专业化/99_代码" || exit 9
PY=/d/Python/python.exe
echo "== report1 $(date)"; $PY -X utf8 pgss_r48_report.py || exit 1
echo "== figures $(date)"; $PY -X utf8 pgss_r48_figures.py || exit 2
echo "== tests $(date)"; $PY -X utf8 pgss_r48_package_tests.py > ../30_R48_图稿与补充材料/package_tests.log 2>&1; rc=$?
$PY -X utf8 -c "
import json; t=json.load(open('../30_R48_图稿与补充材料/R48_PACKAGE_TESTS.json',encoding='utf-8'))
print('全部通过', t['全部通过'])
[print('OK ' if r['通过'] else 'XX ', r['检查'], '|', str(r['说明'])[:160]) for r in t['检查']]"
[ $rc -eq 0 ] || exit 3
echo "== report2 $(date)"; $PY -X utf8 pgss_r48_report.py || exit 4
echo "== package $(date)"; $PY -X utf8 pgss_r48_package.py || exit 5
echo "== done $(date)"
