#!/usr/bin/env bash
# Telegram alert for a failed loop unit. $1 = unit name (from loop-alert@%n).
UNIT="${1:-unknown}"
cd /root/crucible
python3 - "$UNIT" <<'PY'
import sys
sys.path.insert(0, "/root/crucible")
from sdk.notify import telegram_msg
u = sys.argv[1]
telegram_msg(f"🔴 <b>LOOP FAILED</b>: <code>{u}</code> exited non-zero — "
             f"check <code>journalctl -u {u}</code>. Registry: research-wiki/loops.md")
PY
