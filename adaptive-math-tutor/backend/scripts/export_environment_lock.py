from __future__ import annotations

import subprocess
import sys
from pathlib import Path


OUTPUT = Path("requirements-lock.txt")


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stderr.strip())
        return completed.returncode

    OUTPUT.write_text(completed.stdout, encoding="utf-8", newline="\n")
    print(f"Wrote full environment lock: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
