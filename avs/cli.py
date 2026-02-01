from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avs", description="Audio-Visual Synchronizer (AVS)")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("smoke", help="Run smoke checks (see: python -m avs.smoke)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        from avs.smoke import main as smoke_main

        return smoke_main([])

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

