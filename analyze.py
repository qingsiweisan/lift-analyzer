"""
Lift Analyzer - Powerlifting Form Analysis CLI Tool
力量举三大项动作分析工具

Usage:
    python analyze.py <video_path> --type deadlift
    python analyze.py <video_path> --type squat --backend wham
    python analyze.py <video_path> --type bench --barbell-model barbell.pt

Examples:
    python analyze.py my_deadlift.mp4 -t deadlift
    python analyze.py squat_video.mp4 -t squat -o ./results --backend wham
    python analyze.py bench.mp4 -t bench --no-video
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exercises.deadlift import DeadliftAnalyzer
from exercises.squat import SquatAnalyzer
from exercises.bench import BenchAnalyzer


ANALYZERS = {
    "deadlift": DeadliftAnalyzer,
    "squat": SquatAnalyzer,
    "bench": BenchAnalyzer,
}

TYPE_ALIASES = {
    "dl": "deadlift", "硬拉": "deadlift",
    "sq": "squat", "深蹲": "squat",
    "bp": "bench", "卧推": "bench",
}


def resolve_type(name):
    name = name.lower().strip()
    return TYPE_ALIASES.get(name, name)


def main():
    parser = argparse.ArgumentParser(
        description="Lift Analyzer - 力量举三大项动作分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exercise types / 运动类型:
  deadlift (dl, 硬拉)  - Deadlift form analysis
  squat    (sq, 深蹲)  - Squat form analysis
  bench    (bp, 卧推)  - Bench press form analysis

Backends / 姿态估计后端:
  mediapipe  - 2D pose (default, no GPU needed)
  wham       - 3D pose (requires CUDA GPU, higher accuracy)

Examples:
  python analyze.py video.mp4 -t deadlift
  python analyze.py video.mp4 -t 深蹲 --backend wham
  python analyze.py video.mp4 -t bp --barbell-model barbell.pt
        """,
    )
    parser.add_argument("video", help="Path to the video file / 视频文件路径")
    parser.add_argument("-t", "--type", default="deadlift",
                        help="Exercise type: deadlift/squat/bench (default: deadlift)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <video_dir>/<type>_analysis/)")
    parser.add_argument("--backend", default="mediapipe", choices=["mediapipe", "wham"],
                        help="Pose estimation backend (default: mediapipe)")
    parser.add_argument("--barbell-model", default=None,
                        help="Path to YOLOv8 barbell detection model (.pt)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip annotated video generation (faster)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart generation")

    args = parser.parse_args()

    # Validate video path
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Resolve exercise type
    exercise_type = resolve_type(args.type)
    if exercise_type not in ANALYZERS:
        print(f"Error: Unknown exercise type '{args.type}'")
        print(f"Available types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        video_dir = os.path.dirname(os.path.abspath(args.video))
        output_dir = os.path.join(video_dir, f"{exercise_type}_analysis")

    # Run analysis
    analyzer = ANALYZERS[exercise_type]()
    backend_label = "WHAM 3D" if args.backend == "wham" else "MediaPipe 2D"
    print(f"{'='*60}")
    print(f"  Lift Analyzer - {analyzer.exercise_name_cn} ({analyzer.exercise_name})")
    print(f"{'='*60}")
    print(f"  Video:    {args.video}")
    print(f"  Output:   {output_dir}")
    print(f"  Backend:  {backend_label}")
    print(f"  Barbell:  {args.barbell_model or 'wrist approximation'}")
    print(f"  Video:    {'ON' if not args.no_video else 'OFF'}")
    print(f"  Chart:    {'ON' if not args.no_chart else 'OFF'}")
    print(f"{'='*60}")
    print()

    analyzer.run(
        video_path=args.video,
        output_dir=output_dir,
        generate_video=not args.no_video,
        generate_charts=not args.no_chart,
        backend=args.backend,
        barbell_model=args.barbell_model,
    )


if __name__ == "__main__":
    main()
