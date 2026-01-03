#!/usr/bin/env python3
import os, select, subprocess, ctypes, time, signal, sys

IN_MODIFY = 0x00000002
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
EVENT_MASK = IN_MODIFY | IN_CREATE | IN_DELETE

libc = ctypes.CDLL("libc.so.6")
SRC = "/home/ljy/Desktop/RMArmorDetect"
DST = "rry@192.168.50.226:/home/rry/RMArmorDetect"
PORT = "22"

# 需要忽略的目录（相对于 SRC）
# 例如忽略 SRC/runs 和 SRC/logs/tmp
IGNORE_DIRS = [
    "datasets/",
    ".git/",
    "runs/"
]
 
# 预先算好需要忽略的绝对路径，方便后面判断
IGNORE_ABS = {os.path.join(SRC, d) for d in IGNORE_DIRS}

fd = libc.inotify_init()

def add_watch_recursive(root):
    for d, subdirs, _ in os.walk(root):
        # 过滤不需要递归进去的子目录
        subdirs[:] = [
            sd for sd in subdirs
            if os.path.join(d, sd) not in IGNORE_ABS
        ]
        if d in IGNORE_ABS:
            continue
        libc.inotify_add_watch(fd, d.encode(), EVENT_MASK)

GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
last_sync = 0
running = True

def remote_sync(reason):
    global last_sync
    if time.time() - last_sync < 0.5:
        return
    last_sync = time.time()
    print(f"{GREEN}[SYNC]{RESET} Triggered by {reason}")

    cmd = [
        "rsync", "-az", "--inplace",
        "-e", f"ssh -p {PORT}", "--delete",
    ]

    # rsync 同步时也忽略这些目录
    for d in IGNORE_DIRS:
        # 使用相对路径排除：runs/  logs/tmp/
        cmd.extend(["--exclude", d.rstrip("/") + "/"])

    cmd.extend([SRC + "/", DST])

    subprocess.run(cmd)
    print(f"{YELLOW}[DEBUG]{RESET} rsync done {time.strftime('%H:%M:%S')}")

def handle_exit(signum, frame):
    global running
    print(f"\n{RED}[EXIT]{RESET} Caught signal {signum}, cleaning up...")
    try:
        os.close(fd)
    except OSError:
        pass
    running = False

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

print(f"{YELLOW}[DEBUG]{RESET} Monitoring {SRC} ...")
add_watch_recursive(SRC)

while running:
    try:
        r, _, _ = select.select([fd], [], [], 1)
        if fd in r:
            os.read(fd, 4096)
            remote_sync("FILE CHANGE")
    except OSError:
        break

print(f"{GREEN}[INFO]{RESET} Remote sync watcher exited.")
sys.exit(0)
