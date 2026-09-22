"""投篮检测与阶段切分。

思路：投篮出手时手腕会快速上举，用「手腕相对肩部高度的向上速度峰值」
定位出手帧；再围绕出手帧用膝盖角度找到下蹲底，切出
下蹲(dip)/举球(lift)/出手(release)/跟随(follow) 四个阶段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config as C
from .pose import PoseSequence, angle_deg, distance, gradient, median_filter_1d, midpoint


@dataclass
class Shot:
    index: int
    release_frame: int
    release_time: float
    frames: dict = field(default_factory=dict)   # 阶段名 -> (start_frame, end_frame)
    window: tuple = (0, 0)                       # 分析窗口 [start, end]
    side: str = "right"
    detected: bool = True                        # False = 手动指定出手时刻

    def phase(self, name: str) -> Optional[tuple]:
        return self.frames.get(name)

    def to_dict(self, fps: float) -> dict:
        return {
            "index": self.index,
            "release_frame": self.release_frame,
            "release_time_s": round(self.release_time, 3),
            "side": self.side,
            "source": "auto" if self.detected else "manual",
            "phases": {k: {"start_s": round(v[0] / fps, 3), "end_s": round(v[1] / fps, 3),
                           "start_frame": int(v[0]), "end_frame": int(v[1])}
                       for k, v in self.frames.items()},
        }


@dataclass
class ShotDetectionResult:
    shots: list[Shot]
    side: str
    torso_scale: float
    signals: dict          # 逐帧信号，供指标计算与调试
    notes: list[str] = field(default_factory=list)


def pick_shooting_side(seq: PoseSequence, torso_scale: float) -> tuple[str, str]:
    """自动判断投篮手。

    侧面拍摄时投篮手离镜头更近、遮挡更少，因此优先看手腕检测的可视度；
    可视度接近时再看谁举得更高。
    """
    scores = {}
    for side, (sh_idx, wr_idx) in {"left": (C.L_SHOULDER, C.L_WRIST),
                                   "right": (C.R_SHOULDER, C.R_WRIST)}.items():
        vis = seq.visibility(wr_idx)
        vis = vis[seq.valid & np.isfinite(vis)]
        rel = (seq.point(sh_idx)[:, 1] - seq.point(wr_idx)[:, 1]) / torso_scale
        rel = rel[seq.valid & np.isfinite(rel)]
        scores[side] = (
            float(np.median(vis)) if len(vis) else 0.0,
            float(np.percentile(rel, 95)) if len(rel) else -np.inf,
        )
    vis_gap = abs(scores["left"][0] - scores["right"][0])
    if vis_gap >= C.ShotConfig().side_visibility_margin:
        side = "left" if scores["left"][0] > scores["right"][0] else "right"
        reason = "手腕可见度"
    else:
        side = "left" if scores["left"][1] > scores["right"][1] else "right"
        reason = "举高幅度"
    note = (f"自动判定投篮手：{'左' if side == 'left' else '右'}手（依据{reason}；"
            f"可见度 L={scores['left'][0]:.2f} R={scores['right'][0]:.2f}，"
            f"举高 L={scores['left'][1]:.2f} R={scores['right'][1]:.2f}）")
    return side, note


def find_wrist_highs(x: np.ndarray, min_height: float, min_rise: float,
                     rise_frames: int, min_gap: int) -> list[int]:
    """找手腕的显著高点：高度达标，且相对之前窗口内的低谷有足够抬升。

    只用「向前回看的抬升量」而不是两侧凸起度：慢动作视频里手腕在跟随阶段
    会长时间停在高位，两侧凸起度会被抬高，而抬升量不受影响。
    """
    n = len(x)
    cands = []
    for i in range(1, n - 1):
        if not np.isfinite(x[i]) or x[i] < min_height:
            continue
        if x[i] >= np.nanmax(x[max(0, i - 2):i + 3]) and x[i] > x[i - 1]:
            cands.append(i)
    peaks = []
    for i in sorted(cands, key=lambda t: x[t], reverse=True):
        if any(abs(i - p) < min_gap for p in peaks):
            continue
        lo = max(0, i - rise_frames)
        base = x[lo:i + 1]
        if not np.isfinite(base).any():
            continue
        if x[i] - np.nanmin(base) >= min_rise:
            peaks.append(i)
    return sorted(peaks)


def _detect_releases(rel_h: np.ndarray, rel_head: np.ndarray, v: np.ndarray, knee: np.ndarray,
                     elbow: np.ndarray, valid: np.ndarray, quality: np.ndarray, fps: float,
                     cfg: C.ShotConfig) -> tuple[list[int], list[str]]:
    """出手点 = 手腕轨迹（相对肩部高度）的显著高点。

    真实投篮的出手点同时满足：手腕举到肩上明显高度、之前有快速举球、之前有屈膝蓄力、
    出手瞬间肘部已伸展、且肩肘腕关键点检测质量足够。
    这些门槛配合滤掉走路抬手、捡球、接篮板球、画面淡入淡出之类的误报。
    """
    min_gap = int(cfg.min_release_gap_s * fps)
    peaks = find_wrist_highs(rel_h, cfg.release_min_height_ratio, cfg.min_rise_ratio,
                             int(cfg.rise_lookback_s * fps), min_gap)
    releases, rejected = [], []
    dip_win = int(cfg.dip_search_s * fps)
    pre_win = int(cfg.pre_release_search_s * fps)
    last_release = None
    for p in peaks:
        if not valid[p]:
            continue
        if np.isfinite(quality[p]) and quality[p] < cfg.min_joint_visibility:
            rejected.append(f"{p / fps:.1f}s 处手腕高点关键点可见度过低({quality[p]:.2f})")
            continue
        if np.isfinite(elbow[p]) and elbow[p] < cfg.min_elbow_angle_at_release:
            rejected.append(f"{p / fps:.1f}s 处手腕高点时肘部仍折叠({elbow[p]:.0f}°<{cfg.min_elbow_angle_at_release:.0f}°)，球还在手上")
            continue
        # 举球窗口：从下蹲附近到出手点。需要用上一投的位置截断，否则会跨投取到上一投的动作。
        lo = max(0, p - pre_win)
        if last_release is not None:
            lo = max(lo, (last_release + p) // 2)
        # 举球速度：取整个举球阶段的最大向上速度。只看最后 0.6s 会漏掉
        # "快速上升后长时间保持跟随"的投篮（上升发生在更早的时刻）。
        vmax = np.nanmax(v[lo:p + 1]) if np.isfinite(v[lo:p + 1]).any() else -np.inf
        if vmax < cfg.min_upward_velocity:
            rejected.append(f"{p / fps:.1f}s 处手腕高点举球速度不足({vmax:.2f}<{cfg.min_upward_velocity})")
            continue
        # 接球误报：出手点低 且 举球节奏异常（下蹲底到出手过快或过慢）→ 判定为接球而非投篮
        dseg = knee[lo:p + 1]
        if np.isfinite(dseg).any():
            dip_bottom = lo + int(np.nanargmin(dseg))
            lift_s = (p - dip_bottom) / fps
            low_release = np.isfinite(rel_head[p]) and rel_head[p] < cfg.reject_release_height_under
            if low_release and (lift_s < cfg.reject_lift_under_s or lift_s > cfg.reject_lift_over_s):
                rejected.append(f"{p / fps:.1f}s 处手腕高点举球节奏异常({lift_s:.2f}s)"
                                f"且出手点过低({rel_head[p]:.2f})，判定为接球而非投篮")
                continue
        dlo = max(0, p - dip_win)
        kmax = np.nanmax(knee[dlo:p + 1]) if np.isfinite(knee[dlo:p + 1]).any() else np.inf
        kmin = np.nanmin(knee[dlo:p + 1]) if np.isfinite(knee[dlo:p + 1]).any() else np.nan
        if not (np.isfinite(kmin) and kmin <= cfg.knee_dip_max_deg):
            rejected.append(f"{p / fps:.1f}s 处手腕高点前没有屈膝蓄力(最低膝角{kmin:.0f}°>{cfg.knee_dip_max_deg:.0f}°)")
            continue
        releases.append(p)
        last_release = p
    notes = []
    if rejected:
        notes.append("已过滤的手腕高点：" + "；".join(rejected))
    return releases, notes


def _segment_phases(
    release: int,
    knee_angle: np.ndarray,
    wrist_rel: np.ndarray,
    n_frames: int,
    fps: float,
    cfg: C.ShotConfig,
    window_start: int,
    window_end: int,
) -> dict:
    pre = int(cfg.pre_release_search_s * fps)
    lo = max(window_start, release - pre)
    hi = release

    # 下蹲底 = 出手前膝盖屈曲最大（角度最小）的帧
    seg = knee_angle[lo:hi + 1]
    if np.isfinite(seg).any():
        dip_bottom = lo + int(np.nanargmin(seg))
    else:
        dip_bottom = lo

    # 下蹲起点：从下蹲底往回找膝盖最接近伸直的帧
    stand_lo = max(window_start, dip_bottom - int(1.2 * fps))
    seg2 = knee_angle[stand_lo:dip_bottom + 1]
    if np.isfinite(seg2).any():
        dip_start = stand_lo + int(np.nanargmax(seg2))
    else:
        dip_start = stand_lo

    # 跟随结束：手腕回落到肩部以下，或到达窗口上限
    follow_max = min(window_end, release + int(cfg.follow_through_max_s * fps))
    follow_end = follow_max
    for t in range(release + 1, follow_max + 1):
        if np.isfinite(wrist_rel[t]) and wrist_rel[t] < cfg.follow_end_wrist_ratio:
            follow_end = t
            break

    r0 = max(dip_bottom, release - max(1, int(0.05 * fps)))
    r1 = min(follow_max, release + max(2, int(0.10 * fps)))
    return {
        "dip": (dip_start, dip_bottom),
        "lift": (dip_bottom, release),
        "release": (r0, r1),
        "follow": (release, follow_end),
    }


def detect_shots(
    seq: PoseSequence,
    side: str = "auto",
    cfg: Optional[C.ShotConfig] = None,
    manual_releases: Optional[list[float]] = None,
) -> ShotDetectionResult:
    cfg = cfg or C.ShotConfig()
    notes: list[str] = []
    fps = seq.fps

    shoulder = midpoint(seq, C.L_SHOULDER, C.R_SHOULDER)
    hip = midpoint(seq, C.L_HIP, C.R_HIP)
    torso = median_filter_1d(distance(shoulder, hip), 15)
    torso_scale = float(np.nanmedian(torso[np.isfinite(torso) & (torso > 1e-6)])) if np.isfinite(torso).any() else 0.0
    if not torso_scale or torso_scale <= 0:
        return ShotDetectionResult([], "right", 0.0, {}, ["无法计算躯干尺度：姿态数据可能全部无效"])

    if side == "auto":
        side, side_note = pick_shooting_side(seq, torso_scale)
        notes.append(side_note)

    wrist_idx = C.L_WRIST if side == "left" else C.R_WRIST
    shoulder_idx = C.L_SHOULDER if side == "left" else C.R_SHOULDER
    knee_idx = C.L_KNEE if side == "left" else C.R_KNEE
    elbow_idx = C.L_ELBOW if side == "left" else C.R_ELBOW
    hip_idx = C.L_HIP if side == "left" else C.R_HIP
    ankle_idx = C.L_ANKLE if side == "left" else C.R_ANKLE

    wrist = seq.point(wrist_idx)
    shoulder_pt = seq.point(shoulder_idx)
    rel_h = (shoulder_pt[:, 1] - wrist[:, 1]) / torso_scale
    rel_h[~seq.valid] = np.nan
    rel_head = (seq.point(C.NOSE)[:, 1] - wrist[:, 1]) / torso_scale
    rel_head[~seq.valid] = np.nan

    v = gradient(np.nan_to_num(rel_h, nan=0.0), fps)
    v = smooth_vec(v, cfg.velocity_smooth_window)
    v[~seq.valid] = np.nan

    knee = angle_deg(seq.point(hip_idx), seq.point(knee_idx), seq.point(ankle_idx))
    elbow = angle_deg(seq.point(shoulder_idx), seq.point(elbow_idx), seq.point(wrist_idx))

    valid = seq.valid.copy()
    quality = np.minimum.reduce([
        seq.visibility(shoulder_idx), seq.visibility(elbow_idx), seq.visibility(wrist_idx),
    ])
    if manual_releases:
        releases = sorted(int(round(t * fps)) for t in manual_releases if 0 <= t * fps < seq.frame_count)
        notes.append(f"使用手动指定的 {len(releases)} 个出手时刻")
        auto = False
    else:
        releases, reject_notes = _detect_releases(rel_h, rel_head, v, knee, elbow, valid, quality, fps, cfg)
        notes.extend(reject_notes)
        auto = True
        if not releases:
            notes.append("未自动检测到出手时刻，可用 --releases 手动指定出手时间（秒）")

    shots: list[Shot] = []
    for i, r in enumerate(releases):
        prev_r = releases[i - 1] if i > 0 else None
        next_r = releases[i + 1] if i + 1 < len(releases) else None
        w_start = int(prev_r + (r - prev_r) * 0.5) if prev_r is not None else max(0, r - int(1.8 * fps))
        w_end = int(r + (next_r - r) * 0.5) if next_r is not None else min(seq.frame_count - 1, r + int(1.5 * fps))
        phases = _segment_phases(r, knee, rel_h, seq.frame_count, fps, cfg, w_start, w_end)
        shots.append(Shot(
            index=i + 1,
            release_frame=r,
            release_time=r / fps,
            frames=phases,
            window=(w_start, w_end),
            side=side,
            detected=auto,
        ))

    signals = {
        "wrist_rel_height": rel_h,
        "wrist_velocity": v,
        "knee_angle": knee,
        "elbow_angle": elbow,
        "torso_scale": np.full(seq.frame_count, torso_scale),
        "wrist_idx": wrist_idx,
        "shoulder_idx": shoulder_idx,
        "knee_idx": knee_idx,
        "elbow_idx": elbow_idx,
        "hip_idx": hip_idx,
        "ankle_idx": ankle_idx,
    }
    return ShotDetectionResult(shots, side, torso_scale, signals, notes)


def smooth_vec(x: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(x) < window:
        return x
    n = len(x)
    w = window if window % 2 == 1 else window + 1
    half = (w - 1) // 2
    padded = np.pad(x, half, mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")[:n]
