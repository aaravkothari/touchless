"""CLI entry point: python -m touchless <command>."""

import argparse

from . import app
from .config import Config


def main():
    parser = argparse.ArgumentParser(
        prog="touchless",
        description="Control the mouse cursor with your eyes/head via webcam.",
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preview", help="visualize tracking, no cursor control")

    sub.add_parser("calibrate", help="3-posture calibration + accuracy validation")

    p_run = sub.add_parser("run", help="drive the cursor (calibrate first)")
    p_run.add_argument("--click", choices=["off", "dwell", "blink"], default="off",
                       help="click method (default off)")
    p_run.add_argument("--log", metavar="FILE.csv", default=None,
                       help="record features + predictions to a CSV for offline analysis")

    args = parser.parse_args()
    cfg = Config(camera_index=args.camera)

    if args.command == "preview":
        app.preview(cfg)
    elif args.command == "calibrate":
        app.calibrate(cfg)
    elif args.command == "run":
        app.run(cfg, args.click, args.log)


if __name__ == "__main__":
    main()
