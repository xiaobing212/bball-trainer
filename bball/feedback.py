"""规则引擎：把指标翻译成「问题 + 建议」；以及多次投篮的动作一致性评分。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config as C
from .metrics import ShotMetrics


@dataclass
class Issue:
    rule_id: str
    title: str
    severity: str          # warn / info
    detail: str
    advice: str

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "title": self.title, "severity": self.severity,
                "detail": self.detail, "advice": self.advice}


def _f(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"


def evaluate_shot(m: ShotMetrics, cfg: C.FeedbackConfig) -> list[Issue]:
    issues: list[Issue] = []

    if m.knee_dip_deg is not None and m.knee_dip_deg > cfg.knee_dip_good_max:
        issues.append(Issue(
            "knee_insufficient", "下蹲不足", "warn",
            f"下蹲最低点膝盖角约 {m.knee_dip_deg}°（理想 {cfg.knee_dip_good_min:.0f}–{cfg.knee_dip_good_max:.0f}°）",
            "投篮发力从下肢开始，先屈膝下蹲再起。可练「坐椅子」式下蹲：臀部向后下坐，膝盖角降到 110–130° 再出手。"))
    elif m.knee_dip_deg is not None and m.knee_dip_deg < cfg.knee_dip_good_min:
        issues.append(Issue(
            "knee_too_deep", "下蹲过深", "info",
            f"下蹲最低点膝盖角约 {m.knee_dip_deg}°（理想 {cfg.knee_dip_good_min:.0f}–{cfg.knee_dip_good_max:.0f}°）",
            "下蹲过深会让出手变慢、还原变难。尝试减小下蹲幅度，保持节奏紧凑。"))

    if m.release_height_rel_head is not None and m.release_height_rel_head < cfg.release_height_low:
        issues.append(Issue(
            "release_low", "出手点偏低", "warn",
            f"出手瞬间手腕低于头顶（相对鼻尖 {m.release_height_rel_head:+.2f} 个躯干长度）",
            "出手点越高越难被封盖。把球举到额头以上再出手，出手时手臂尽量向上伸展、在头上方释放。"))

    if m.elbow_release_deg is not None and m.elbow_release_deg < cfg.elbow_release_min:
        issues.append(Issue(
            "elbow_bent_release", "出手时肘部未伸展", "warn",
            f"出手瞬间肘角约 {m.elbow_release_deg}°（建议 ≥ {cfg.elbow_release_min:.0f}°）",
            "出手瞬间手臂应接近伸直（推投）。练习近距离单手向上推球，体会「向上伸臂、手腕下压」的发力次序。"))

    if m.elbow_set_deg is not None and m.elbow_set_deg < cfg.elbow_set_good_min:
        issues.append(Issue(
            "elbow_overfolded", "蓄力时肘部过度折叠", "info",
            f"举球蓄力阶段肘角最小约 {m.elbow_set_deg}°（建议 ≥ {cfg.elbow_set_good_min:.0f}°）",
            "肘部折叠过紧会把球压到身后，出手轨迹变长。保持球在肩前上方，肘部约 60–90°。"))

    if m.release_angle_deg is not None:
        if m.release_angle_deg < cfg.release_angle_min:
            issues.append(Issue(
                "release_angle_flat", "出手弧线偏平", "warn",
                f"出手角度约 {m.release_angle_deg}°（建议 {cfg.release_angle_min:.0f}–{cfg.release_angle_max:.0f}°）",
                "弧线太平会缩小入筐窗口。出手时更多向上发力，让球先向上抛再下落。"))
        elif m.release_angle_deg > cfg.release_angle_max:
            issues.append(Issue(
                "release_angle_high", "出手弧线过高", "info",
                f"出手角度约 {m.release_angle_deg}°（建议 {cfg.release_angle_min:.0f}–{cfg.release_angle_max:.0f}°）",
                "弧线过高对力量要求大、稳定性差。适度降低出手角度，让球更平稳。"))

    if m.body_lean_deg is not None:
        if m.body_lean_deg < cfg.body_lean_back_max:
            issues.append(Issue(
                "lean_back", "出手时身体过度后仰", "warn",
                f"出手瞬间躯干后仰约 {abs(m.body_lean_deg):.0f}°",
                "后仰会让出手点前移、力量传导中断。保持躯干竖直、臀部在脚跟上方，起跳向上而不是向后。"))
        elif m.body_lean_deg > cfg.body_lean_forward_max:
            issues.append(Issue(
                "lean_forward", "出手时身体过度前倾", "warn",
                f"出手瞬间躯干前倾约 {m.body_lean_deg:.0f}°",
                "前倾容易导致力量向前卸掉。出手时保持身体在垂直线上，落地在起跳点附近。"))

    if m.max_trunk_lean_deg is not None and m.max_trunk_lean_deg > cfg.trunk_lean_max:
        issues.append(Issue(
            "trunk_fold", "下蹲时上身前倾过多", "warn",
            f"下蹲到出手的过程中躯干最多前倾约 {m.max_trunk_lean_deg:.0f}°"
            f"（建议 ≤ {cfg.trunk_lean_max:.0f}°；出手瞬间是 {_f(m.body_lean_deg, '°')}）",
            "屈膝是对的，但同时把上半身对折下去会切断「下肢→手臂」的力量传导，出手也会忽高忽低。"
            "练「坐椅子」式的下蹲：臀部往后下坐、胸口保持朝前，上身基本竖直地降下去再起来。"))

    if m.follow_through_s is not None and m.follow_through_s < cfg.follow_through_min_s:
        issues.append(Issue(
            "follow_incomplete", "跟随动作不完整", "warn",
            f"出手后手保持上方仅约 {m.follow_through_s:.2f}s（建议 ≥ {cfg.follow_through_min_s:.2f}s）",
            "出手后手腕应保持「压腕下垂」姿势停留一下再放下。练习时口中数「1」再收手，强化跟随定型。"))

    if m.lift_duration_s is not None:
        if m.lift_duration_s < cfg.lift_duration_min_s:
            issues.append(Issue(
                "lift_too_fast", "举球发力仓促", "info",
                f"从下蹲底到出手仅 {m.lift_duration_s:.2f}s",
                "举球节奏过快会导致全身发力不同步。可以先慢后快：下蹲到位后稍作停顿，再顺畅起跳出手。"))
        elif m.lift_duration_s > cfg.lift_duration_max_s:
            issues.append(Issue(
                "lift_too_slow", "出手前停顿过长", "info",
                f"从下蹲底到出手约 {m.lift_duration_s:.2f}s",
                "下蹲与出手之间的停顿会损失弹性势能，也让防守有时间反应。让下蹲到出手连成一条顺畅的弧线。"))

    return issues


@dataclass
class Consistency:
    overall: Optional[float]
    items: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"overall_score": self.overall, "items": self.items}


CONSISTENCY_LABELS = {
    "release_height": "出手高度",
    "release_angle": "出手角度",
    "elbow_release": "出手时肘角",
    "knee_dip": "下蹲深度",
    "lift_duration": "举球节奏",
}


def compute_consistency(metrics: list[ShotMetrics], cfg: C.FeedbackConfig) -> Consistency:
    values = {
        "release_height": [m.release_height_rel_shoulder for m in metrics],
        "release_angle": [m.release_angle_deg for m in metrics],
        "elbow_release": [m.elbow_release_deg for m in metrics],
        "knee_dip": [m.knee_dip_deg for m in metrics],
        "lift_duration": [m.lift_duration_s for m in metrics],
    }
    items = []
    weighted_sum = 0.0
    weight_total = 0.0
    for key, vals in values.items():
        arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        w = cfg.consistency_weights.get(key, 0.0)
        if len(arr) < 2 or abs(arr.mean()) < 1e-9:
            items.append({"key": key, "label": CONSISTENCY_LABELS[key], "mean": None, "std": None,
                          "cv": None, "score": None, "weight": w})
            continue
        cv = float(arr.std(ddof=1) / abs(arr.mean()))
        score = float(np.clip(1.0 - cv / cfg.consistency_cv_tolerance, 0.0, 1.0) * 100)
        items.append({"key": key, "label": CONSISTENCY_LABELS[key],
                      "mean": round(float(arr.mean()), 2), "std": round(float(arr.std(ddof=1)), 2),
                      "cv": round(cv, 3), "score": round(score, 1), "weight": w})
        weighted_sum += score * w
        weight_total += w

    overall = round(weighted_sum / weight_total, 1) if weight_total > 0 else None
    return Consistency(overall, items)


def summarize(metrics: list[ShotMetrics], shot_issues: dict[int, list[Issue]], cfg: C.FeedbackConfig) -> dict:
    counts = Counter()
    for issues in shot_issues.values():
        for it in issues:
            counts[it.rule_id] += 1
    titles = {}
    advice = {}
    order = []
    for idx in sorted(shot_issues):
        for it in shot_issues[idx]:
            titles[it.rule_id] = it.title
            if it.rule_id not in advice:
                advice[it.rule_id] = it.advice
            if it.rule_id not in order:
                order.append(it.rule_id)
    total = len(metrics)
    top = []
    for rid in sorted(order, key=lambda r: -counts[r]):
        top.append({
            "rule_id": rid,
            "title": titles[rid],
            "count": counts[rid],
            "total_shots": total,
            "advice": advice[rid],
        })
    return {"issue_counts": dict(counts), "top_issues": top}
