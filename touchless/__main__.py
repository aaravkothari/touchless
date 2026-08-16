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

    p_prev = sub.add_parser("preview", help="visualize tracking, no cursor control")
    p_prev.add_argument("--input", choices=["gaze", "hand", "hand-wrist"],
                        default="gaze", help="what to track (default gaze)")

    sub.add_parser("calibrate",
                   help="pursuit calibration (follow the moving cursor) + validation")

    sub.add_parser("retrain",
                   help="refit the model from the recorded pursuit session (no camera)")

    p_run = sub.add_parser("run", help="drive the cursor")
    p_run.add_argument("--input", choices=["gaze", "hand", "hand-wrist"],
                       default="gaze",
                       help="gaze = eyes+head (calibrate first); "
                            "hand = right index finger, no calibration needed; "
                            "hand-wrist = index finger relative to the palm "
                            "(moving the whole arm/hand doesn't move the cursor)")
    p_run.add_argument("--click", choices=["off", "dwell", "blink", "pinch"],
                       default="off",
                       help="click method (default off; pinch is hand-mode only)")
    p_run.add_argument("--log", metavar="FILE.csv", default=None,
                       help="record per-frame signals to a CSV (gaze: features"
                            " + predictions; hand: pointer, gate state,"
                            " cursor) - for offline stability analysis")

    args = parser.parse_args()
    cfg = Config(camera_index=args.camera)

    if args.command == "preview":
        app.preview(cfg, args.input)
    elif args.command == "calibrate":
        app.calibrate(cfg)
    elif args.command == "retrain":
        app.retrain(cfg)
    elif args.command == "run":
        app.run(cfg, args.click, args.log, args.input)


if __name__ == "__main__":
    main()
