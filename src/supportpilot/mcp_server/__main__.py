"""python -m supportpilot.mcp_server [--http] [--port 8765]"""

import argparse
import logging
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportPilot MCP server")
    parser.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # MCP clients launch servers from their own working directory; settings (.env, data/, models/)
    # are relative to the project root, so move there before anything reads them.
    root = Path(os.environ.get("SUPPORTPILOT_HOME") or Path(__file__).resolve().parents[3])
    os.chdir(root)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Qdrant call otherwise

    from supportpilot.db.base import init_db
    from supportpilot.mcp_server.server import mcp

    init_db()
    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()  # stdio: stdout carries the protocol, logs go to stderr


if __name__ == "__main__":
    main()
