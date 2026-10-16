#!/bin/sh
# R47 收尾链：报告 → 图 → 包测试 → 报告（含 A4）→ 打包自检
cd "D:/Research/2_课题/PGSS_表型引导选择性专业化/99_代码" || exit 9
PY=/d/Python/python.exe
echo "== report1 $(date)"; $PY pgss_r47_report.py || exit 1
echo "== figures $(date)"; $PY pgss_r47_figures.py || exit 2
echo "== tests $(date)"; $PY pgss_r47_package_tests.py > ../29_R47_写作交接/package_tests.log 2>&1; rc=$?; tail -60 ../29_R47_写作交接/package_tests.log
[ $rc -eq 0 ] || exit 3
echo "== report2 $(date)"; $PY pgss_r47_report.py || exit 4
echo "== package $(date)"; $PY pgss_r47_package.py || exit 5
echo "== done $(date)"
