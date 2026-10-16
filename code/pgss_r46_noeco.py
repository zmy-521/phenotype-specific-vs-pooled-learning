"""R46：对 T2 的 python 进程关闭 EcoQoS（执行速度节流），只改调度提示，不改计算。
用法：python noeco.py once   或   python noeco.py loop <哨兵目录>（每 20 秒扫一次新进程，见 _T2_DONE/_T2_FAILED 退出）"""
import ctypes, ctypes.wintypes as W, subprocess, sys, time, pathlib, datetime

k = ctypes.WinDLL("kernel32", use_last_error=True)
k.OpenProcess.restype = W.HANDLE


class PPT(ctypes.Structure):
    _fields_ = [("Version", W.ULONG), ("ControlMask", W.ULONG), ("StateMask", W.ULONG)]


EXEC_SPEED = 0x1
ProcessPowerThrottling = 4
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED = 0x1000


def t2_pids():
    """命令行含 pgss_r46_t2 / _r46_t2_driver 的 python 进程。"""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, errors="replace").stdout
    procs = {}
    for line in out.splitlines():
        if "\t" in line:
            pid, cmd = line.split("\t", 1)
            procs[int(pid)] = cmd
    roots = {p for p, c in procs.items() if "r46_t2" in c}
    # multiprocessing spawn 子进程：命令行带 parent_pid=<T2 进程>
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
    mode = sys.argv[1]
    done = set()
    sentinel = pathlib.Path(sys.argv[2]) if mode == "loop" else None
    while True:
        new = [p for p in t2_pids() if p not in done]
        res = {p: opt_out(p) for p in new}
        done |= {p for p, ok in res.items() if ok}
        if res:
            print(datetime.datetime.now().strftime("%m-%d %H:%M:%S"), res, flush=True)
        if mode != "loop":
            return
        if (sentinel / "_T2_DONE").exists() or (sentinel / "_T2_FAILED").exists():
            print("sentinel seen, exit", flush=True)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
