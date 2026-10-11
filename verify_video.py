# verify_video.py
# Run a local prediction on one or more videos, without starting the server.
#
# Usage (from the project folder):
#   python verify_video.py sample.mp4
#   python verify_video.py real1.mp4 fake1.mp4 --expect real fake
#
# --expect is optional. Give one label per video ("real" or "fake") and the
# script reports whether each prediction matched.

import argparse
import os
import sys
import time

from inference import load_model, predict_video

DEFAULT_CKPT = os.environ.get("DEEPFAKE_CKPT", "checkpoints/finetune_resume.pt")
THRESHOLD = 0.5
UNCERTAIN_BAND = 0.10   # probabilities within 0.4 to 0.6 are flagged as uncertain


def main():
    parser = argparse.ArgumentParser(description="Verify the Deepfake Shield model on local videos.")
    parser.add_argument("videos", nargs="+", help="Video file(s) to analyse")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to the .pt checkpoint")
    parser.add_argument("--expect", nargs="+", choices=["real", "fake"],
                        help="Known true label for each video, in the same order")
    args = parser.parse_args()

    if args.expect and len(args.expect) != len(args.videos):
        sys.exit(f"--expect needs {len(args.videos)} labels, got {len(args.expect)}")

    for v in args.videos:
        if not os.path.isfile(v):
            sys.exit(f"File not found: {v}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint} (run this from the project folder)")

    print("Loading model...")
    model = load_model(args.checkpoint)
    print()

    results = []
    for i, path in enumerate(args.videos):
        name = os.path.basename(path)
        print(f"[{i + 1}/{len(args.videos)}] {name}")
        t0 = time.time()
        try:
            p = predict_video(model, path)
        except ValueError as e:
            print(f"  ERROR: {e}\n")
            results.append((name, None, None, None))
            continue

        label = "fake" if p > THRESHOLD else "real"
        flag = ""
        if abs(p - THRESHOLD) < UNCERTAIN_BAND:
            flag = "  (uncertain: close to the 0.5 threshold)"
        match = ""
        if args.expect:
            ok = label == args.expect[i]
            match = "  MATCH" if ok else "  MISMATCH"

        print(f"  fake probability: {p:.4f}")
        print(f"  predicted label:  {label}{flag}")
        if args.expect:
            print(f"  expected label:   {args.expect[i]}{match}")
        print(f"  time: {time.time() - t0:.1f}s\n")
        results.append((name, p, label, args.expect[i] if args.expect else None))

    print("=" * 60)
    print(f"{'video':<30}{'prob':>8}  {'pred':<6}{'expected':<10}")
    for name, p, label, exp in results:
        if p is None:
            print(f"{name:<30}{'error':>8}")
        else:
            print(f"{name:<30}{p:>8.4f}  {label:<6}{(exp or '-'):<10}")
    print("=" * 60)

    if args.expect:
        scored = [(l, e) for _, p, l, e in results if p is not None]
        if scored:
            correct = sum(1 for l, e in scored if l == e)
            print(f"Correct: {correct}/{len(scored)}")


if __name__ == "__main__":
    main()