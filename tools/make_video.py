"""Create a simple MP4 from an image and optional audio."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Make a simple mp4 from image + audio")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--audio", default=None, help="Optional audio path")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration if audio missing")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.audio and os.path.exists(args.audio):
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            args.image,
            "-i",
            args.audio,
            "-shortest",
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            args.out,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            args.image,
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl=stereo",
            "-t",
            str(args.duration),
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            args.out,
        ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
