"""Commands: simulate, check, detect, eval, train, run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fall_recog.evaluate import (
    clear_alarm_check,
    evaluate_dir,
    protocol_failures,
    render_boundary,
    render_report,
    write_events,
    write_report,
)
from fall_recog.io_csv import CsvError, iter_recordings, read_frames
from fall_recog.rules import FallDetector, load_params
from fall_recog.simulate import generate
from fall_recog.plot import plot_recording
from fall_recog.train import train_dir

ROOT = Path(__file__).resolve().parent
DEFAULT_PARAMS = ROOT / "params.json"
SYNTHETIC = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"

SYNTHETIC_NOTE = (
    "本报告来自合成数据管线自检，用来确认程序、文件格式和事件接口。"
    "表里的数字不是实板跌倒识别准确率。实板 CSV 放到 data/board 后，用同一条 eval 命令重跑。"
)


def _params(path: str) -> dict:
    return load_params(Path(path))


def cmd_simulate(args: argparse.Namespace) -> int:
    written = generate(Path(args.out), seed=args.seed)
    print(f"写出 {len(written)} 段合成记录到 {args.out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    files = iter_recordings(Path(args.input))
    if not files:
        print(f"没有符合命名的 CSV: {args.input}", file=sys.stderr)
        return 1
    failed = 0
    for path in files:
        try:
            frames, warnings = read_frames(path)
        except CsvError as exc:
            failed += 1
            print(f"FAIL {path.name}: {exc}")
            continue
        warning_text = f"；{'；'.join(warnings)}" if warnings else ""
        print(f"OK {path.name} 帧数 {len(frames)} 时长 {frames[-1].t_ms} ms{warning_text}")
    return 1 if failed else 0


def cmd_detect(args: argparse.Namespace) -> int:
    params = _params(args.params)
    files = iter_recordings(Path(args.input))
    if args.split != "all":
        files = [path for path in files if path.stem.endswith(f"_{args.split}")]
    if not files:
        print(f"没有要识别的 CSV: {args.input}", file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    detector = FallDetector(params)
    for path in files:
        frames, warnings = read_frames(path)
        events = detector.run(frames)
        result = {"path": path, "events": events}
        write_events(result, out_dir)
        for event in events:
            print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        for warning in warnings:
            print(f"# warn {path.name}: {warning}", file=sys.stderr)
    print(f"事件已写入 {out_dir}", file=sys.stderr)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    params = _params(args.params)
    out = Path(args.out)
    summary = train_dir(
        Path(args.input),
        out,
        invalid_value=int(params["invalid_value"]),
        vel_fall=float(params["vel_fall"]),
    )
    print(f"训练窗 {summary['samples']} 条，训练集拟合 {summary['train_acc']:.3f}")
    print("各类窗数:", summary["counts"])
    print(f"模型写入 {summary['path']}")
    print("合成数据上的拟合率不能当作实板准确率。")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    csv_path = Path(args.input)
    out = Path(args.out) if args.out else csv_path.with_suffix(".png")
    written = plot_recording(csv_path, out)
    print(f"距离门幅度图 {written}")
    print("纵轴是距离门，不是人体离地高度。")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    params = _params(args.params)
    source = Path(args.input)
    files_root = source
    results = evaluate_dir(files_root, params, Path(args.out) / "events")
    if args.split != "all":
        results = [item for item in results if item["path"].stem.endswith(f"_{args.split}")]
    if not results:
        print(f"没有要评测的 CSV: {args.input}", file=sys.stderr)
        return 1
    title = args.title or "跌倒识别测试结果"
    text = render_report(title, results, params, args.note or "数据目录里的记录。")
    report_path = Path(args.out) / "测试结果.md"
    write_report(report_path, text)
    failures = protocol_failures(results)
    print(text)
    print(f"报告已写入 {report_path}")
    if args.strict and failures:
        print("严格模式未通过:", file=sys.stderr)
        for message in failures:
            print(f"- {message}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    params = _params(args.params)
    written = generate(SYNTHETIC, seed=args.seed)
    print(f"写出 {len(written)} 段合成记录")
    model_path = ROOT / params.get("model_path", "models/action_rf.joblib")
    if params.get("use_classifier"):
        summary = train_dir(
            SYNTHETIC / "train",
            model_path,
            invalid_value=int(params["invalid_value"]),
            vel_fall=float(params["vel_fall"]),
        )
        print(f"用合成 train 训练 {summary['samples']} 窗 -> {summary['path']}")
        params = _params(args.params)

    test_results = evaluate_dir(SYNTHETIC / "test", params, RESULTS / "events" / "test")
    train_results = evaluate_dir(SYNTHETIC / "train", params, RESULTS / "events" / "train")
    boundary_results = evaluate_dir(SYNTHETIC / "boundary", params, RESULTS / "events" / "boundary")

    test_text = render_report("规范动作测试集（合成，P01）", test_results, params, SYNTHETIC_NOTE)
    train_text = render_report("规范动作训练集（合成，P02、P03）", train_results, params, SYNTHETIC_NOTE)
    boundary_text = render_boundary(boundary_results)

    fall_frames, _ = read_frames(next(item["path"] for item in test_results if item["label"] == "fall"))
    cleared_ok, cleared_note = clear_alarm_check(params, fall_frames)
    clear_line = f"## 解除报警\n\n{cleared_note}\n"
    test_text = test_text.rstrip() + "\n\n" + clear_line

    write_report(RESULTS / "测试结果_test.md", test_text)
    write_report(RESULTS / "测试结果_train.md", train_text)
    write_report(RESULTS / "规则边界.md", boundary_text)

    example = next(item for item in test_results if item["label"] == "fall")
    example_path = RESULTS / "event_example.jsonl"
    example_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in example["events"]) + "\n",
        encoding="utf-8",
    )

    print(test_text)
    print(boundary_text)
    print(f"测试报告 {RESULTS / '测试结果_test.md'}")
    print(f"训练报告 {RESULTS / '测试结果_train.md'}")
    print(f"边界报告 {RESULTS / '规则边界.md'}")
    print(f"示例事件 {example_path}")

    failures = protocol_failures(test_results) + protocol_failures(train_results) + protocol_failures(boundary_results)
    if not cleared_ok:
        failures.append(cleared_note)
    if failures:
        print("自检未通过:", file=sys.stderr)
        for message in failures:
            print(f"- {message}", file=sys.stderr)
        return 1
    print("合成数据自检通过。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fall_recog", description="雷达记录上的跌倒规则识别")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    sub = parser.add_subparsers(dest="cmd", required=True)

    simulate = sub.add_parser("simulate", help="生成合成 CSV")
    simulate.add_argument("--out", default=str(SYNTHETIC))
    simulate.add_argument("--seed", type=int, default=20261029)
    simulate.set_defaults(func=cmd_simulate)

    check = sub.add_parser("check", help="检查 CSV 是否符合数据接口")
    check.add_argument("input")
    check.set_defaults(func=cmd_check)

    detect = sub.add_parser("detect", help="识别并把事件写成 JSONL")
    detect.add_argument("input")
    detect.add_argument("--out", default=str(RESULTS / "events"))
    detect.add_argument("--split", choices=("train", "test", "all"), default="all")
    detect.set_defaults(func=cmd_detect)

    evaluate = sub.add_parser("eval", help="按动作统计报警、疑似和正常")
    evaluate.add_argument("--input", default=str(SYNTHETIC / "test"))
    evaluate.add_argument("--out", default=str(RESULTS))
    evaluate.add_argument("--split", choices=("train", "test", "all"), default="all")
    evaluate.add_argument("--title", default="")
    evaluate.add_argument("--note", default="")
    evaluate.add_argument("--strict", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    train = sub.add_parser("train", help="用标注 CSV 训练动作分类器")
    train.add_argument("--input", default=str(SYNTHETIC / "train"))
    train.add_argument("--out", default=str(ROOT / "models" / "action_rf.joblib"))
    train.set_defaults(func=cmd_train)

    plot = sub.add_parser("plot", help="画出 20 个距离门的 I/Q 幅度随时间变化")
    plot.add_argument("input")
    plot.add_argument("--out", default="")
    plot.set_defaults(func=cmd_plot)

    run = sub.add_parser("run", help="生成合成数据、训练、识别、写测试报告")
    run.add_argument("--seed", type=int, default=20261029)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CsvError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
