from __future__ import annotations

import argparse
import asyncio

from qec.operator.irc_server import IRCServer


def _is_local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local deterministic QEC IRC server core.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=6667, type=int, help="Bind port (default: 6667)")
    parser.add_argument("--max-clients", default=32, type=int, help="Maximum concurrent clients")
    parser.add_argument("--allow-nonlocal-bind", action="store_true", help="Allow non-local bind host")
    return parser


async def _run(args: argparse.Namespace) -> None:
    server = IRCServer(host=args.host, port=args.port, max_clients=args.max_clients)
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not _is_local_host(args.host) and not args.allow_nonlocal_bind:
        parser.error("non-local bind requires --allow-nonlocal-bind")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
