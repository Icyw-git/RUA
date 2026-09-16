"""Record child exit status and bounded diagnostics outside the GPU process."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from bootstrap import CODE, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["debug", "pilot", "replay"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base = ROOT / "artifacts/wla-stage1"
    out = Path(tempfile.mkdtemp(prefix=f"supervisor-{args.mode}-", dir=base))
    report = dict(status="starting", mode=args.mode, supervisor_pid=os.getpid(),
                  started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    child = None

    def save():
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "report.json")

    save()
    print(json.dumps({"supervisor_directory": str(out)}), flush=True)
    with (out / "child.log").open("w") as output:
        child = subprocess.Popen(
            ["bash", str(CODE / "scripts/run_stage1.sh"), args.mode, *args.arguments],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1"),
        )
        report.update(status="running", child_pid=child.pid)
        save()
        code = child.wait()
    report.update(status="completed" if code == 0 else "failed", returncode=code,
                  child_signal=-code if code < 0 else None,
                  ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save()
    print(json.dumps(report), flush=True)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
