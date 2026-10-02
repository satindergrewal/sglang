#!/bin/bash
# mimo-serve — start/stop/restart/status/logs for the 8015 daily serve.
#
# This is the native-weights long-context serve the user confirmed working
# (client name "mimo"): original checkpoint, fp8 KV cache, 1,048,576-token
# pool, tool-call parsing + alias. NOT the 4-bit-KV variant (chat breaks on
# it — see boot_nvfp4_8015.sh). The EXL3-quantized alternative is boot_lb6.sh.
#
# Usage: /mnt/nvme0/work-exl3/mimo-serve.sh {start|stop|restart|status|logs}
set -u
BASE=/mnt/nvme0/work-exl3
NAME=exl-native
PORT=8015
BOOT=$BASE/boot_fp8kv_1m.sh

is_running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }

wait_healthy() {
    echo "waiting for health on :$PORT (weights load ~3-4 min)..."
    for i in $(seq 1 90); do
        sleep 10
        CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)
        if [ "$CODE" = "200" ]; then
            echo "HEALTH_OK (~$((i * 10))s)"
            docker logs "$NAME" 2>&1 | grep -E "max_total_num_tokens=" | tail -1
            return 0
        fi
        if docker ps -a --format '{{.Names}}' | grep -qx "$NAME" && ! is_running; then
            echo "CONTAINER_DIED during boot — last errors:"
            docker logs "$NAME" 2>&1 | grep -iE "error|OutOfMemory" | tail -5
            return 1
        fi
        # container not created yet: boot script's docker run still starting
    done
    echo "TIMEOUT waiting for health (container may still be loading — check: $0 status)"
    return 1
}

case "${1:-status}" in
    start)
        if is_running; then
            echo "already running:"; docker ps --filter name="$NAME" --format '{{.Names}} ({{.Status}})'
        else
            echo "starting $NAME from $BOOT"
            setsid nohup bash "$BOOT" < /dev/null > /tmp/mimo_serve_boot.log 2>&1 &
            wait_healthy
        fi
        ;;
    stop)
        if is_running || docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
            docker rm -f "$NAME" >/dev/null 2>&1
            echo "stopped ($NAME removed)"
        else
            echo "not running"
        fi
        ;;
    restart)
        "$0" stop
        sleep 2
        "$0" start
        ;;
    status)
        if is_running; then
            echo "running: $NAME ($(docker ps --filter name="$NAME" --format '{{.Status}}'))"
            CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)
            echo "health: $CODE"
            docker logs "$NAME" 2>&1 | grep -E "max_total_num_tokens=" | tail -1
        elif docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
            echo "EXITED (crashed or stopped):"
            docker logs "$NAME" 2>&1 | tail -5
        else
            echo "not running (no container)"
        fi
        ;;
    logs)
        docker logs -f --tail 100 "$NAME" 2>&1
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
