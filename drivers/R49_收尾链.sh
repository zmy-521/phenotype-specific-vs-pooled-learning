#!/bin/sh
# R49 收尾链：报告（写 xlsx）→ 包测试 → 报告（含 A7）→ 打包自检
cd "D:/Research/2_课题/PGSS_表型引导选择性专业化/99_代码" || exit 9
PY=/d/Python/python.exe
$PY -X utf8 pgss_r49_report.py || exit 1
$PY -X utf8 pgss_r49_tests.py > ../31_R49_只读补齐/package_tests.log 2>&1 || { tail -40 ../31_R49_只读补齐/package_tests.log; exit 2; }
tail -1 ../31_R49_只读补齐/package_tests.log
$PY -X utf8 pgss_r49_report.py || exit 3
$PY -X utf8 pgss_r49_package.py || exit 4
