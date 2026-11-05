"""Score recordings and write the action-by-action test report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fall_recog.io_csv import CsvError, iter_recordings, read_frames, read_header_meta, read_sidecar
from fall_recog.rules import FallDetector, apply_uplink
from fall_recog.schema import LABEL_ZH, LABELS, Frame

ROOT = Path(__file__).resolve().parent


def classify_events(events: list[dict]) -> str:
    """Use the state left at the end of the recording.

    A suspect that later recovers, or an alarm cleared by no_human, is normal.
    Passing through a state is not the result.
    """
    if not events:
        return "normal"
    state = events[-1]["state"]
    if state in ("alarm", "suspect"):
        return state
    return "normal"


def expected_outcome(label: str, sidecar: dict[str, str]) -> str:
    forced = sidecar.get("expect", "")
    if forced in ("alarm", "no_alarm"):
        return forced
    return "alarm" if label == "fall" else "no_alarm"


def outcome_ok(kind: str, expect: str) -> bool:
    if expect == "alarm":
        return kind == "alarm"
    return kind != "alarm"


def _peak(frames: list[Frame], invalid: int) -> int:
    speeds = [
        abs(frame.vel)
        for frame in frames
        if frame.vel != invalid and frame.rng != invalid
    ]
    if not speeds:
        return 0
    return max(speeds)


def score_file(path: Path, params: dict) -> dict:
    frames, warnings = read_frames(path)
    header = read_header_meta(path)
    sidecar = read_sidecar(path)
    events = FallDetector(params).run(frames)
    kind = classify_events(events)
    label = frames[0].label
    expect = expected_outcome(label, sidecar)
    alarm_at = None
    if kind == "alarm":
        alarm_at = next(event["t_ms"] for event in events if event["state"] == "alarm")
    suspect_at = next((event["t_ms"] for event in events if event["state"] == "suspect"), None)
    onset_text = sidecar.get("motion_onset_ms", "")
    onset = int(onset_text) if onset_text.isdigit() else None
    delay = None if alarm_at is None or onset is None else alarm_at - onset
    return {
        "path": path,
        "label": label,
        "kind": kind,
        "expect": expect,
        "ok": outcome_ok(kind, expect),
        "peak": _peak(frames, int(params["invalid_value"])),
        "frames": len(frames),
        "duration_ms": frames[-1].t_ms,
        "alarm_at": alarm_at,
        "suspect_at": suspect_at,
        "onset": onset,
        "delay": delay,
        "events": events,
        "warnings": warnings,
        "header": header,
        "sidecar": sidecar,
        "note": sidecar.get("note", ""),
    }


def write_events(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in result["events"]]
    (out_dir / f"{result['path'].stem}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def render_report(title: str, results: list[dict], params: dict, source_note: str) -> str:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_label[result["label"]].append(result)

    lines = [
        f"# {title}",
        "",
        source_note,
        "",
        "判定看记录结束时的状态。中途进过疑似或报警、最后回到正常的，记为正常。结束时仍是疑似、没有升级，记为「仅疑似」。",
        "",
        "## 参数",
        "",
    ]
    if params.get("use_classifier"):
        lines.extend(
            [
                f"- 分类器：滑窗 {params.get('window_ms', 2000)} ms，I/Q 幅度为主，gear/rng/vel/snr 为辅，P(fall) ≥ {params.get('fall_proba', 0.45)} 视为跌倒样",
                f"- 模型 {params.get('model_path', 'models/action_rf.joblib')}",
                f"- 丢掉 rng 或 vel 等于 {params.get('invalid_value', -999)} 的帧",
                f"- 静止 |vel| ≤ {params['still_vel']}，{params['still_suspect_ms']} ms 进入疑似，{params['still_alarm_ms']} ms 升级为报警",
                "- 合成训练只能自检程序；实板准确率必须用板端 I/Q 重训",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- 丢掉 rng 或 vel 等于 {params.get('invalid_value', -999)} 的帧",
                f"- 快速冲击 |vel| ≥ {params['vel_fall']}（单位与 CSV 的 vel 相同，当前假定 {params.get('assumed_vel_unit', '未填')}）",
                f"- 冲击需在 {params['impact_span_ms']} ms 内至少出现 {params['min_impact_frames']} 帧，且距离沿速度方向变化 ≥ {params['impact_rng']}（假定 {params.get('assumed_rng_unit', '未填')}）",
                f"- 距离回到冲击前 {params['return_rng']} 以内，或从落点向冲击前回升 ≥ {params['rebound_rng']}，取消冲击。沿冲击方向继续移动不取消",
                f"- 静止 |vel| ≤ {params['still_vel']}（不要求贴在落点 {params['return_rng']} 以内）",
                f"- 静止 {params['still_suspect_ms']} ms 进入疑似，{params['still_alarm_ms']} ms 升级为报警",
                f"- 帧里有 hgt 时，还要相对冲击前下降 ≥ {params['height_drop']}，且当前高度 ≤ {params['height_lie_max']}（假定 {params.get('assumed_hgt_unit', '未填')}）。没有这一列时不查高度",
                "",
            ]
        )
    lines.extend(
        [
            "## 分动作结果",
            "",
            "| 动作 | 段数 | 报警 | 仅疑似 | 正常 | 最大\\|vel\\|中位数 | 不符合预期 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        group = by_label.get(label, [])
        if not group:
            continue
        alarm = sum(1 for item in group if item["kind"] == "alarm")
        suspect = sum(1 for item in group if item["kind"] == "suspect")
        normal = sum(1 for item in group if item["kind"] == "normal")
        bad = sum(1 for item in group if not item["ok"])
        lines.append(
            f"| {LABEL_ZH[label]} ({label}) | {len(group)} | {alarm} | {suspect} | {normal} | {_median([item['peak'] for item in group])} | {bad} |"
        )

    fall_bad = sum(1 for item in by_label.get("fall", []) if not item["ok"])
    other = [item for item in results if item["label"] != "fall"]
    other_alarm = sum(1 for item in other if item["kind"] == "alarm")
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 跌倒段数 {len(by_label.get('fall', []))}，其中不符合预期 {fall_bad}",
            f"- 非跌倒段数 {len(other)}，其中发出报警 {other_alarm}",
            "",
            "## 跌倒报警延迟",
            "",
            "延迟 = 报警时间 − 记录表里的动作开始时间。其中包含倒下过程，以及静止确认时间。",
            "",
            "| 文件 | 动作开始 ms | 报警 ms | 延迟 ms |",
            "|---|---:|---:|---:|",
        ]
    )
    delays = []
    for item in by_label.get("fall", []):
        if item["delay"] is None:
            lines.append(f"| {item['path'].name} | {item['onset'] if item['onset'] is not None else ''} |  |  |")
            continue
        delays.append(item["delay"])
        lines.append(f"| {item['path'].name} | {item['onset']} | {item['alarm_at']} | {item['delay']} |")
    if delays:
        lines.append("")
        lines.append(f"平均延迟 {sum(delays) // len(delays)} ms。")

    failed = [item for item in results if not item["ok"]]
    lines.extend(["", "## 不符合预期的记录", ""])
    if not failed:
        lines.append("没有。")
    else:
        for item in failed:
            lines.append(
                f"- {item['path'].name}：标注 {item['label']}，结果 {item['kind']}，预期 {item['expect']}。{item['note']}"
            )

    warned = [item for item in results if item["warnings"]]
    if warned:
        lines.extend(["", "## 数据警告", ""])
        for item in warned:
            lines.append(f"- {item['path'].name}：{'；'.join(item['warnings'])}")
    lines.append("")
    return "\n".join(lines)


def render_boundary(results: list[dict]) -> str:
    lines = [
        "# 规则边界",
        "",
        "这些样本不进入规范动作的成绩。快速坐下、缓慢倒下、快走急停，单靠速度门限和位移门限分不开。",
        "",
        "有 `hgt` 时，坐下停在坐姿、急停保持站立，都不会报警。距离从落点向冲击前回升，也会取消。没有 `hgt`、距离也不回升时，这两条仍会报警。",
        "",
        "缓慢倒下不报警。它和主动躺下都是速度到不了跌倒门限、位移够大、然后停在低处。放宽 `vel_fall` 会把主动躺下一起报出来。",
        "",
        "| 文件 | 标注 | 结果 | 预期现象 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for item in results:
        phenomenon = "出现报警" if item["kind"] == "alarm" else "没有报警"
        lines.append(
            f"| {item['path'].name} | {LABEL_ZH[item['label']]} | {phenomenon} | {item['expect']} | {item['note']} |"
        )
    lines.extend(
        [
            "",
            "## 还不能当成实板结果",
            "",
            "- `data/board` 是空的，距离和速度的单位还没有在实板上核对。",
            "- 合成按厘米、厘米每秒。`vel_fall=90`、`impact_rng=30` 换到实板可能差一个数量级。",
            "- 检测不用 `snr`，也不用 `gear`。过 1 米、2 米档时实板 `rng` 可能跳变，这里没有处理。",
            "",
        ]
    )
    return "\n".join(lines)


def clear_alarm_check(params: dict, fall_frames: list[Frame]) -> tuple[bool, str]:
    detector = FallDetector(params)
    events = detector.run(fall_frames)
    if not any(event["state"] == "alarm" for event in events):
        return False, "样本没有进入报警，无法检查解除。"
    last = fall_frames[-1]
    extra: list[dict] = []
    for step in range(1, 11):
        frame = Frame(
            t_ms=last.t_ms + step * 50,
            frame_id=last.frame_id + step,
            gear=last.gear,
            rng=last.rng,
            vel=0,
            snr=last.snr,
            human=1,
            lb=last.lb,
            ub=last.ub,
            label="fall",
            hgt=last.hgt,
        )
        extra.extend(detector.push(frame))
    if any(event["state"] == "alarm" for event in extra):
        return False, "报警之后静止帧又发出了一次报警。"
    cleared = apply_uplink(detector, {"device_id": params["device_id"], "cmd": "clear"}, frame)
    if cleared["state"] != "normal" or cleared["reason"] != "cleared":
        return False, "解除命令没有回到 normal/cleared。"
    extra_after: list[dict] = []
    for step in range(1, 21):
        stayed = Frame(
            t_ms=frame.t_ms + step * 50,
            frame_id=frame.frame_id + step,
            gear=frame.gear,
            rng=frame.rng,
            vel=1,
            snr=frame.snr,
            human=1,
            lb=frame.lb,
            ub=frame.ub,
            label="fall",
            hgt=frame.hgt,
        )
        extra_after.extend(detector.push(stayed))
    if any(event["state"] == "alarm" for event in extra_after):
        return False, "解除之后人仍然躺着，却再次报警。"
    return True, "报警锁存；解除后回到 normal；人继续躺着不会立刻再次报警。"


def evaluate_dir(path: Path, params: dict, event_dir: Path | None = None) -> list[dict]:
    files = iter_recordings(path)
    if not files:
        raise CsvError(f"{path} 里没有符合命名的 CSV")
    results = []
    for csv_path in files:
        result = score_file(csv_path, params)
        results.append(result)
        if event_dir is not None:
            write_events(result, event_dir)
    return results


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def protocol_failures(results: list[dict]) -> list[str]:
    messages = []
    for item in results:
        if item["ok"]:
            continue
        messages.append(f"{item['path'].name} 标注 {item['label']} 结果 {item['kind']} 预期 {item['expect']}")
    return messages
