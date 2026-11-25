"""Allow ``python -m geneware`` to launch the application.

By default this starts the browser dashboard (Streamlit). Pass
``--textual`` to run the legacy terminal dashboard instead. Subcommands
(``parse``, ``scan``, ``bench``, ``analyze``, ``report``, ``gpu-status``)
are forwarded to the argparse CLI.

The browser dashboard binds port 8501 by default. If the port is already
in use the launcher stops with a clear message instead of silently
attaching to a stale process; pass ``--port <N>`` to use another port.
"""

import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

_WEBAPP = Path(__file__).resolve().parent / "webapp.py"
_CLI_COMMANDS = {"parse", "scan", "bench", "analyze", "report",
                 "gpu-status", "--help", "-h"}
_DEFAULT_PORT = 8501


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _is_streamlit_app(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def _open_browser_when_ready(url: str, port: int,
                             timeout: float = 30.0) -> None:
    """Wait for Streamlit's health endpoint, then open the URL.

    Runs in a daemon thread so it never blocks shutdown. If the server
    never becomes healthy the browser is simply not opened.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/healthz", timeout=1
            ) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.4)
    else:
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _launch_webapp(port: int) -> int:
    url = f"http://localhost:{port}"
    if _port_in_use(port):
        if _is_streamlit_app(port):
            print(
                f"Port {port} is already serving a Streamlit app at {url}.\n"
                "Opening it in your browser. To start a fresh GeneWeaver "
                "instance instead, stop the existing process first, or "
                f"use: python -m geneware --port {port + 1}",
                file=sys.stderr,
            )
            webbrowser.open(url)
        else:
            print(
                f"Port {port} is already in use by another process, so "
                f"GeneWeaver cannot start at {url}.\n"
                f"Free the port or use another one: "
                f"python -m geneware --port {port + 1}",
                file=sys.stderr,
            )
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(_WEBAPP),
        "--server.headless", "true",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    print(f"Starting GeneWeaver browser app at {url} …", flush=True)
    print("The app will open in your browser automatically. "
          "Press Ctrl+C to stop.", flush=True)
    threading.Thread(
        target=_open_browser_when_ready, args=(url, port), daemon=True
    ).start()
    return subprocess.call(cmd)


def main() -> int:
    argv = sys.argv[1:]
    if "--textual" in argv:
        from .app import main as textual_main
        return textual_main()
    if argv and argv[0] in _CLI_COMMANDS:
        from .cli import main as cli_main
        return cli_main(argv)
    port = _DEFAULT_PORT
    if "--port" in argv:
        idx = argv.index("--port")
        try:
            port = int(argv[idx + 1])
        except (IndexError, ValueError):
            print("Usage: python -m geneware [--port N]", file=sys.stderr)
            return 2
    # Default: browser dashboard.
    return _launch_webapp(port)


if __name__ == "__main__":
    sys.exit(main())
