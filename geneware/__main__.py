"""Allow ``python -m geneware`` to launch the application.

By default this starts the browser dashboard (Streamlit). Pass
``--textual`` to run the legacy terminal dashboard instead. Subcommands
(``parse``, ``scan``, ``bench``, ``analyze``, ``report``, ``gpu-status``)
are forwarded to the argparse CLI.
"""

import subprocess
import sys
from pathlib import Path

_WEBAPP = Path(__file__).resolve().parent / "webapp.py"
_CLI_COMMANDS = {"parse", "scan", "bench", "analyze", "report",
                 "gpu-status", "--help", "-h"}


def main() -> int:
    argv = sys.argv[1:]
    if "--textual" in argv:
        from .app import main as textual_main
        return textual_main()
    if argv and argv[0] in _CLI_COMMANDS:
        from .cli import main as cli_main
        return cli_main(argv)
    # Default: browser dashboard.
    cmd = [sys.executable, "-m", "streamlit", "run", str(_WEBAPP),
           "--server.headless", "true"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
