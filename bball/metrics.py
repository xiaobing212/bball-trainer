"""每次投篮的生物力学指标计算（2D 近似，侧面 90° 机位最可靠）。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import config as C
from .pose import PoseSequence, angle_deg, midpoint
from .shots import Shot, ShotDetectionResult


def _val(x: float, ndigits: int = 1) -> Optional[float]:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


@dataclass
class ShotMetrics:
    shot_index: int
    release_time_s: float
    # 膝盖
    knee_standing_deg: Optional[float] = None
    knee_dip_deg: Optional[float] = None          # 下蹲最低点膝盖角
    knee_release_deg: Optional[float] = None
    # 手肘
    elbow_set_deg: Optional[float] = None         # 蓄力时最小肘角(set point)
    elbow_release_deg: Optional[float] = None     # 出手瞬间肘角
    # 高度（躯干长度为单位）
    release_height_rel_head: Optional[float] = None      # 手腕相对鼻尖, 正=高于鼻尖
    release_height_rel_shoulder: Optional[float] = None
    set_point_height_rel_shoulder: Optional[float] = None
    dip_depth_rel: Optional[float] = None                # 下蹲时髋部下沉幅度
    jump_height_rel: Optional[float] = None              # 起跳高度(相对身高)
    # 角度
    release_angle_deg: Optional[float] = None     # 手腕速度方向与水平夹角
    body_lean_deg: Optional[float] = None         # 出手时身体前后倾, 正=前倾朝篮筐
    max_trunk_lean_deg: Optional[float] = None    # 下蹲→出手全程躯干最大前倾角（只看出手瞬间会漏掉「对折下身」）
    # 跟随
    follow_peak_rel: Optional[float] = None       # 出手后手腕最高点(相对肩)
    follow_through_s: Optional[float] = None      # 出手后手保持肩以上的时长
    # 节奏
    dip_duration_s: Optional[float] = None
    lift_duration_s: Optional[float] = None
    # 元信息
    release_px: Optional[tuple] = None            # 归一化坐标, 供渲染

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_facing(seq: PoseSequence) -> int:
    """判断球员面朝画面的哪一侧：+1 朝右, -1 朝左。用脚尖相对脚踝的方向估计。"""
    toe = midpoint(seq, C.L_FOOT, C.R_FOOT)
    ankle = midpoint(seq, C.L_ANKLE, C.R_ANKLE)
    d = (toe[:, 0] - ankle[:, 0])
    d = d[seq.valid & np.isfinite(d)]
    if len(d) == 0 or np.nanmedian(d) == 0:
        # 兜底：鼻子相对髋部的朝向
        nose = seq.point(C.NOSE)
        hip = midpoint(seq, C.L_HIP, C.R_HIP)
        d2 = (nose[:, 0] - hip[:, 0])
        d2 = d2[seq.valid & np.isfinite(d2)]
        if len(d2) == 0 or np.nanmedian(d2) == 0:
            return 1
        return 1 if np.nanmedian(d2) > 0 else -1
    return 1 if np.nanmedian(d) > 0 else -1


def compute_frame_series(seq: PoseSequence, det: ShotDetectionResult) -> dict:
    """逐帧衍生信号（写入 metrics.csv）。"""
    s = det.signals
    facing = estimate_facing(seq)
    shoulder = midpoint(seq, C.L_SHOULDER, C.R_SHOULDER)
    hip = midpoint(seq, C.L_HIP, C.R_HIP)
    ankle = midpoint(seq, C.L_ANKLE, C.R_ANKLE)
    torso = s["torso_scale"]
    body_lean = np.degrees(np.arctan2((shoulder[:, 0] - hip[:, 0]) * facing,
                                      -(shoulder[:, 1] - hip[:, 1] + 1e-9)))
    return {
        "elbow_angle_deg": s["elbow_angle"],
        "knee_angle_deg": s["knee_angle"],
        "wrist_rel_shoulder": s["wrist_rel_height"],
        "wrist_velocity": s["wrist_velocity"],
        "body_lean_deg": body_lean,
        "hip_height_rel": (ankle[:, 1] - hip[:, 1]) / torso,
    }


def compute_shot_metrics(shot: Shot, seq: PoseSequence, det: ShotDetectionResult) -> ShotMetrics:
    s = det.signals
    fps = seq.fps
    r = shot.release_frame
    torso = det.torso_scale
    facing = estimate_facing(seq)

    knee = s["knee_angle"]
    elbow = s["elbow_angle"]
    rel_h = s["wrist_rel_height"]

    dip = shot.frames["dip"]
    lift = shot.frames["lift"]
    follow = shot.frames["follow"]

    m = ShotMetrics(shot_index=shot.index, release_time_s=round(shot.release_time, 2))

    def seg(a, b):
        return slice(max(0, a), min(seq.frame_count - 1, b) + 1)

    def nanmin(x):
        return np.nanmin(x) if np.isfinite(x).any() else np.nan

    def nanmax(x):
        return np.nanmax(x) if np.isfinite(x).any() else np.nan

    # 膝盖
    m.knee_standing_deg = _val(knee[dip[0]])
    m.knee_dip_deg = _val(nanmin(knee[seg(*dip)]))
    m.knee_release_deg = _val(knee[r])

    # 手肘
    set_seg = seg(lift[0], lift[1])
    elbow_lift = elbow[set_seg]
    if np.isfinite(elbow_lift).any():
        set_i = int(np.nanargmin(elbow_lift)) + set_seg.start
    else:
        set_i = lift[0]
    m.elbow_set_deg = _val(elbow[set_i])
    m.elbow_release_deg = _val(elbow[r])

    # 高度：手腕相对鼻尖 / 肩
    nose_y = seq.point(C.NOSE)[r, 1]
    wrist = seq.point(s["wrist_idx"])
    shoulder_pt = seq.point(s["shoulder_idx"])
    m.release_height_rel_head = _val((nose_y - wrist[r, 1]) / torso, 2)
    m.release_height_rel_shoulder = _val(rel_h[r], 2)
    m.set_point_height_rel_shoulder = _val(rel_h[set_i], 2)

    # 下蹲深度：髋部从站直到下蹲底的下沉量
    hip_y = midpoint(seq, C.L_HIP, C.R_HIP)[:, 1]
    m.dip_depth_rel = _val((hip_y[dip[1]] - hip_y[dip[0]]) / torso, 2)

    # 起跳高度：以脚踝上升量 / 站直身高 计
    ankle_y = midpoint(seq, C.L_ANKLE, C.R_ANKLE)[:, 1]
    body_px = ankle_y[dip[0]] - seq.point(C.NOSE)[dip[0], 1]
    if np.isfinite(body_px) and body_px > 1e-6:
        m.jump_height_rel = _val((ankle_y[dip[0]] - ankle_y[r]) / body_px, 3)

    # 出手角度：出手点(手腕最高处)竖直速度接近零，取到达出手点前仍在举球的窗口方向。
    # 注：对比过"速度最快时刻"和"抬升 50% 处"等替代定义，三者差异在噪声范围内，
    # 这个定义最简洁且无缺失值。手腕最高点晚于真实离手时，读数会有偏差（个别投篮的已知噪声）。
    i0 = None
    for win_s in (0.15, 0.25, 0.40):
        k = max(1, int(win_s * fps))
        j = max(0, r - k)
        if wrist[r, 1] - wrist[j, 1] < 0:   # 图像 y 减小 = 向上
            i0 = j
            break
    if i0 is None:
        i0 = max(0, r - max(1, int(0.15 * fps)))
    dt = (r - i0) / fps
    if dt > 0:
        vx = (wrist[r, 0] - wrist[i0, 0]) / dt
        vy = (wrist[r, 1] - wrist[i0, 1]) / dt
        m.release_angle_deg = _val(np.degrees(np.arctan2(abs(vy), abs(vx) + 1e-9)))

    # 身体倾斜
    shoulder_c = midpoint(seq, C.L_SHOULDER, C.R_SHOULDER)
    hip_c = midpoint(seq, C.L_HIP, C.R_HIP)
    dy = -(shoulder_c[r, 1] - hip_c[r, 1])
    dx = (shoulder_c[r, 0] - hip_c[r, 0]) * facing
    _lean_release = np.degrees(np.arctan2(dx, dy + 1e-9))
    m.body_lean_deg = _val(_lean_release) if abs(_lean_release) <= 90 else None   # 超过 90° = 跟丢翻转

    # 下蹲到出手全程的最大躯干前倾。出手瞬间上身往往已经回正，只看那一刻会漏掉
    # 「蹲下去的同时把上半身对折」这类动作（实测：某投出手瞬间 0.4°、过程中 46.6°）。
    # 跟丢时肩/髋中点会翻转成 ~180°，那是跟踪错误不是动作，直接丢弃。
    lean_all = np.degrees(np.arctan2((shoulder_c[:, 0] - hip_c[:, 0]) * facing,
                                     -(shoulder_c[:, 1] - hip_c[:, 1]) + 1e-9))
    # 再压一道中值滤波：真实的「对折」会持续很多帧，跟丢造成的假尖峰只有一两帧
    lean_all = np.array([np.nanmedian(lean_all[max(0, i - 2):i + 3]) for i in range(len(lean_all))])
    lean_seg = lean_all[seg(dip[0], r)]
    lean_seg = np.where(np.abs(lean_seg) <= 80.0, lean_seg, np.nan)
    m.max_trunk_lean_deg = _val(nanmax(lean_seg))

    # 跟随动作
    fseg = seg(follow[0] + 1, follow[1])
    m.follow_peak_rel = _val(nanmax(rel_h[fseg]), 2)
    first_low = None
    for t in range(follow[0] + 1, follow[1] + 1):
        if np.isfinite(rel_h[t]) and rel_h[t] < 0.05:
            first_low = t
            break
    if first_low is not None:
        m.follow_through_s = _val((first_low - r) / fps, 2)
    else:
        m.follow_through_s = _val((follow[1] - r) / fps, 2)

    # 节奏
    m.dip_duration_s = _val((dip[1] - dip[0]) / fps, 2)
    m.lift_duration_s = _val((lift[1] - lift[0]) / fps, 2)

    m.release_px = (float(wrist[r, 0]), float(wrist[r, 1]))
    return m
