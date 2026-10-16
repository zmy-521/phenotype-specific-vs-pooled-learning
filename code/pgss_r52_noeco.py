"""R52：对分支运行的 python 进程关闭 EcoQoS（执行速度节流），只改调度提示，不改计算（同 pgss_r46_noeco.py，进程匹配改为 r52）。
用法：python pgss_r52_noeco.py loop <哨兵目录>（每 20 秒扫一次新进程，见 _R52_DONE/_R52_FAILED 退出）"""
import ctypes, ctypes.wintypes as W, subprocess, sys, time, pathlib, datetime

k = ctypes.WinDLL("kernel32", use_last_error=True)
k.OpenProcess.restype = W.HANDLE


class PPT(ctypes.Structure):
    _fields_ = [("Version", W.ULONG), ("ControlMask", W.ULONG), ("StateMask", W.ULONG)]


EXEC_SPEED = 0x1
ProcessPowerThrottling = 4
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED = 0x1000


def pids():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, errors="replace").stdout
    procs = {}
    for line in out.splitlines():
        if "\t" in line:
            pid, cmd = line.split("\t", 1)
            procs[int(pid)] = cmd
    roots = {p for p, c in procs.items() if "r52" in c and "noeco" not in c}
    return sorted(roots | {p for p, c in procs.items() if any(f"parent_pid={r}," in c for r in roots)})


def opt_out(pid):
    h = k.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED, False, pid)
    if not h:
        return False
    s = PPT(1, EXEC_SPEED, 0)
    ok = bool(k.SetProcessInformation(h, ProcessPowerThrottling, ctypes.byref(s), ctypes.sizeof(s)))
    k.CloseHandle(h)
    return ok


def main():
    sentinel = pathlib.Path(sys.argv[2])
    done = set()
    while True:
        new = [p for p in pids() if p not in done]
        res = {p: opt_out(p) for p in new}
        done |= {p for p, ok in res.items() if ok}
        if res:
            print(datetime.datetime.now().strftime("%m-%d %H:%M:%S"), res, flush=True)
        if (sentinel / "_R52_DONE").exists() or (sentinel / "_R52_FAILED").exists():
            print("sentinel seen, exit", flush=True)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
