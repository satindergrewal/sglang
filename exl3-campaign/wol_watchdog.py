#!/usr/bin/env python3
"""WoL watchdog for MonkeyAMD: send magic packets every 30s; when SSH comes
back, boot the 3.75 daily serve on 8015 and verify coherence. Runs until the
serve is confirmed up or 8h elapse."""
import socket, subprocess, time, datetime

MAC = "58:11:22:ad:e6:70"
HOST = "192.168.0.101"
PKT = b"\xff" * 6 + bytes.fromhex(MAC.replace(":", "")) * 16
LOG = "/home/satinder/.zcode/wol_watchdog.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")
    print(msg, flush=True)

def send_wol():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    for dst, port in (("192.168.0.255", 9), ("192.168.0.101", 9)):
        try:
            s.sendto(PKT, (dst, port))
        except OSError:
            pass
    s.close()

def ssh(cmd, timeout=15):
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes",
         f"satinder@{HOST}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )

def main():
    deadline = time.time() + 8 * 3600
    log("watchdog start (WoL every 30s until SSH returns)")
    booted = False
    while time.time() < deadline:
        try:
            if not booted:
                send_wol()
                r = ssh("echo SSH-BACK")
                if "SSH-BACK" in r.stdout:
                    log("SSH IS BACK — waiting 90s for boot settle, then checking GPUs")
                    time.sleep(90)
                    g = ssh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=60)
                    log(f"gpus after cold boot: {g.stdout.strip()!r}")
                    # push the fixed boot scripts (multimodal opt-out) before booting
                    subprocess.run(
                        ["scp", "-o", "ConnectTimeout=10",
                         "/home/satinder/eaglefix-build/boots/boot_daily375.sh",
                         "/home/satinder/eaglefix-build/boots/boot_eagle28p.sh",
                         f"satinder@{HOST}:/mnt/nvme0/work-exl3/"],
                        capture_output=True, text=True, timeout=60,
                    )
                    ssh("chmod +x /mnt/nvme0/work-exl3/boot_daily375.sh /mnt/nvme0/work-exl3/boot_eagle28p.sh", timeout=30)
                    ssh("nohup /mnt/nvme0/work-exl3/boot_daily375.sh > /dev/null 2>&1 & echo booted-daily", timeout=30)
                    log("fixed boot scripts pushed; daily boot launched")
                    booted = True
            else:
                r = ssh(
                    "docker logs exl3-375 2>&1 | grep -cE 'max_total_num_tokens|fired up'",
                    timeout=30,
                )
                n = "".join(c for c in r.stdout.strip() if c.isdigit())
                if n and int(n) >= 2:
                    log("DAILY SERVE IS UP (boot markers found in exl3-375 logs)")
                    log("watchdog DONE")
                    return
        except Exception as e:  # noqa: BLE001 — keep the watchdog alive
            log(f"transient error: {e!r}")
        time.sleep(30)
    log("watchdog timed out after 8h")

if __name__ == "__main__":
    main()
