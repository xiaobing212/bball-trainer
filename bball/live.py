"""实时投篮分析：把逐帧姿态流变成投篮事件。

设计：实时层只负责"什么时候可能出手了"（轻量状态机 + 环形缓冲）；
一旦事件发生，把前后几秒的缓冲切成一个小窗口，交给现有的离线分析管线
（detect_shots + compute_shot_metrics + evaluate_shot），保证实时结果与报告一致。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sys
from typing import Optional

import numpy as np

from . import config as C
from . import feedback
from .metrics import ShotMetrics, compute_shot_metrics
from .pose import PoseSequence, fill_gaps, smooth
from .shots import detect_shots


@dataclass
class LiveConfig:
    buffer_seconds: float = 6.0      # 环形缓冲时长
    window_pre_s: float = 3.0        # 分析窗口：出手点前取多久
    window_post_s: float = 0.6       # 出手点后取多久。判断跟随动作只需看到 0.5s：
                                     # 手一直举着=好跟随，0.5s 内掉下来=跟随不完整
    min_buffer_s: float = 2.0        # 至少攒够这么久才开始判断
    arm_velocity: float = 0.7        # 触发"手臂上举"的向上速度(躯干/秒)
    arm_height: float = 0.10         # 触发时手腕需高于肩部(躯干比例)
    v_reversal: float = 0.15         # 速度降到该值以下并持续 confirm_frames 帧，视为到达出手点
    confirm_frames: int = 2          # 到达出手点的确认帧数（越小反馈越快，抗噪差些）
    fall_ratio: float = 0.25         # 兜底：从峰值回落该幅度也视为这次上举结束
    fall_velocity: float = -0.8      # 兜底：向下速度低于该值也视为结束
    min_shot_gap_s: float = 2.0      # 两次投篮事件的最小间隔
    velocity_ema: float = 0.5        # 实时速度平滑系数


@dataclass
class LiveShot:
    release_time: float                  # 相对会话开始时长的秒数
    metrics: ShotMetrics
    issues: list
    shot_number: int
    peak_frame: int = 0                  # 出手点所在的帧号（用于实测反馈延迟）

    @property
    def top_issue(self) -> Optional[str]:
        warns = [i for i in self.issues if i.severity == "warn"]
        if warns:
            return warns[0].title
        return self.issues[0].title if self.issues else None


class LiveAnalyzer:
    """逐帧喂入姿态，产出投篮事件。"""

    def __init__(self, fps: float, side: str = "auto", cfg: Optional[C.Config] = None,
                 live_cfg: Optional[LiveConfig] = None):
        self.fps = fps
        self.side = side
        self.cfg = cfg or C.Config()
        self.lcfg = live_cfg or LiveConfig()
        self.frames: deque = deque(maxlen=int(self.lcfg.buffer_seconds * fps))
        self.detected: deque = deque(maxlen=int(self.lcfg.buffer_seconds * fps))
        # 每帧的时间戳（和 frames 一一对应）。出手时刻用它算，不再用「帧号 ÷ 帧率」——
        # 帧率估计会被 measureFps 实时修正，一改「同一个帧号」就被换算成不同的秒数，
        # 实测会出现时刻漂移、甚至倒退（2026-09-24 球场：第 11 投 124.01s、第 12 投 74.12s）。
        self.times: deque = deque(maxlen=int(self.lcfg.buffer_seconds * fps))
        self._t_start: Optional[float] = None    # 这一场第一帧的时间（时间原点）
        self._t_last: Optional[float] = None     # 最近一帧的时间（已含补偿）
        self._t_raw_last: Optional[float] = None  # 最近一帧的**原始**时间戳（未补偿）
        self._t_offset = 0.0                     # 时间戳倒退时的补偿（见 _time_for）
        self.torso_scale: Optional[float] = None
        self._v_ema = 0.0
        self._slow_frames = 0
        self._prev_rel_h: Optional[float] = None
        self.state = "idle"            # idle / rising
        self._peak_rel_h = -np.inf
        self._peak_idx = 0
        self._pending_peak_idx: Optional[int] = None
        self._last_shot_idx = -10**9
        self._last_peak_idx = -10**9     # 上一次分析过的峰值（用于截断窗口，避免跨投取景）
        self._side_locked: Optional[str] = None
        self.shot_count = 0
        self.frame_idx = 0

    # ---------- 对外接口 ----------

    def reset_session(self):
        """开始新一次训练：把时间基准归零。

        出手时刻 = 这一帧的时间戳 - 本场第一帧的时间戳（`self._t_start`），
        不归零的话时间基准会是「程序启动」而不是「按下开始训练」。
        """
        self.frames.clear()
        self.detected.clear()
        self.times.clear()
        self._t_start = None
        self._t_last = None
        self._t_raw_last = None
        self._t_offset = 0.0
        self.frame_idx = 0
        self.state = "idle"
        self._pending_peak_idx = None
        self._peak_rel_h = -np.inf
        self._peak_idx = 0
        self._slow_frames = 0
        self._prev_rel_h = None
        self._v_ema = 0.0
        self._last_shot_idx = -10**9
        self._last_peak_idx = -10**9
        self.shot_count = 0

    @property
    def status(self) -> str:
        if self._pending_peak_idx is not None:
            return "分析中"
        if self.state == "rising":
            return "检测到举球"
        return "等待投篮"

    def _time_for(self, idx: int, t: Optional[float]) -> float:
        """这一帧的时间戳。

        手机端传的是 `video.currentTime`（真实时间）——出手时刻直接用它算，所以
        卡片上的秒数和录屏/离线报告天然对得上。没传的（离线回放、测试）按帧率外推，
        和以前的行为一致。
        两个兜底：① 切摄像头会让 currentTime 归零 → 用偏移量接上，保证单调；
        ② 同一会话里混着传/不传也不会跳（不传时接着上一帧外推）。
        """
        # 注意：比较和累加都只用**原始**时间戳（_t_raw_last）。用补偿后的值去比，
        # 偏移量会被每一帧重复累加、指数级放大（实测 3.8s 会涨到 1.9e34）。
        raw = t if t is not None else (
            self._t_raw_last + 1.0 / self.fps if self._t_raw_last is not None else idx / self.fps)
        if self._t_raw_last is not None and raw < self._t_raw_last:
            self._t_offset += self._t_raw_last - raw + 1.0 / self.fps
            print(f"[live] 时间戳倒退（{self._t_raw_last:.2f} → {raw:.2f}），"
                  f"补偿 {self._t_offset:.2f}s", file=sys.stderr)
        self._t_raw_last = raw
        self._t_last = raw + self._t_offset
        if self._t_start is None:
            self._t_start = self._t_last
        return self._t_last

    def push(self, landmarks: Optional[np.ndarray],
             t: Optional[float] = None) -> Optional[LiveShot]:
        """喂入一帧关键点 (33,4)，返回本帧完成的投篮事件（如果有）。

        t：这一帧的时间戳（秒）。手机端传 `video.currentTime`；离线/测试不传。
        """
        idx = self.frame_idx
        self.frame_idx += 1
        self.times.append(self._time_for(idx, t))
        if landmarks is None:
            self.frames.append(np.full((33, 4), np.nan, dtype=np.float32))
            self.detected.append(False)
            self._prev_rel_h = None
            return None
        self.frames.append(np.asarray(landmarks, dtype=np.float32))
        self.detected.append(True)

        rel_h = self._rel_height(landmarks)
        event = None
        if rel_h is not None:
            v = self._velocity(rel_h)
            event = self._step(idx, rel_h, v)

        # 事件发生后，等窗口攒满再做分析
        if (event is None and self._pending_peak_idx is not None
                and idx - self._pending_peak_idx >= int(self.lcfg.window_post_s * self.fps)):
            event = self._analyze(self._pending_peak_idx)
            self._pending_peak_idx = None
        return event

    # ---------- 内部 ----------

    def _rel_height(self, lm: np.ndarray) -> Optional[float]:
        """手腕相对肩部的高度（躯干长度为单位）。"""
        side = self._side_locked or ("right" if self.side == "auto" else self.side)
        wr, sh = (C.L_WRIST, C.L_SHOULDER) if side == "left" else (C.R_WRIST, C.R_SHOULDER)
        hip_l, hip_r = lm[C.L_HIP], lm[C.R_HIP]
        sh_l, sh_r = lm[C.L_SHOULDER], lm[C.R_SHOULDER]
        if not (np.isfinite(lm[wr, :2]).all() and np.isfinite(lm[sh, :2]).all()):
            return None
        torso = np.linalg.norm((sh_l[:2] + sh_r[:2]) / 2 - (hip_l[:2] + hip_r[:2]) / 2)
        if torso < 1e-6:
            return None
        self.torso_scale = float(torso) if self.torso_scale is None else 0.95 * self.torso_scale + 0.05 * float(torso)
        return (lm[sh, 1] - lm[wr, 1]) / self.torso_scale

    def _velocity(self, rel_h: float) -> float:
        if self._prev_rel_h is None:
            self._prev_rel_h = rel_h
            return 0.0
        dv = (rel_h - self._prev_rel_h) * self.fps
        self._prev_rel_h = rel_h
        a = self.lcfg.velocity_ema
        self._v_ema = a * dv + (1 - a) * self._v_ema
        return self._v_ema

    def _step(self, idx: int, rel_h: float, v: float) -> Optional[LiveShot]:
        lc = self.lcfg
        if self.state == "idle":
            if (v >= lc.arm_velocity and rel_h >= lc.arm_height
                    and idx - self._last_shot_idx >= int(lc.min_shot_gap_s * self.fps)):
                self.state = "rising"
                self._peak_rel_h = rel_h
                self._peak_idx = idx
            return None

        # state == rising
        if rel_h > self._peak_rel_h:
            self._peak_rel_h = rel_h
            self._peak_idx = idx
            self._slow_frames = 0
        else:
            self._slow_frames = self._slow_frames + 1 if v <= lc.v_reversal else 0

        # 出手点确认：速度明显回落并持续几帧（不用等手落下，反馈更快）；
        # 兜底：手快速下落或从峰值大幅回落（应对速度噪声）。
        settled = (self._slow_frames >= lc.confirm_frames
                   and idx > self._peak_idx + 1)
        fallen = (rel_h < self._peak_rel_h - lc.fall_ratio) or (v <= lc.fall_velocity)
        if settled or fallen:
            self.state = "idle"
            self._pending_peak_idx = self._peak_idx
        return None

    def flush(self) -> Optional[LiveShot]:
        """流结束时处理还没分析完的出手（跟随阶段样本会短一些）。"""
        if self._pending_peak_idx is not None:
            shot = self._analyze(self._pending_peak_idx)
            self._pending_peak_idx = None
            return shot
        return None

    def _analyze(self, peak_idx: int) -> Optional[LiveShot]:
        lc = self.lcfg
        frames = list(self.frames)
        det_all = list(self.detected)
        buf_start = self.frame_idx - len(frames)   # 环形缓冲里第一帧的绝对序号
        lo = max(buf_start, peak_idx - int(lc.window_pre_s * self.fps))
        if self._last_peak_idx > 0:
            # 窗口不跨过上一次分析过的峰值，否则两次出手会在间隔抑制里互相压制
            lo = max(lo, (self._last_peak_idx + peak_idx) // 2)
        hi = min(self.frame_idx - 1, peak_idx + int(lc.window_post_s * self.fps))
        if lo >= peak_idx or hi - peak_idx < max(3, int(0.45 * self.fps)):
            return None
        self._last_peak_idx = peak_idx
        window = np.stack(frames[lo - buf_start:hi - buf_start + 1])
        det = np.array(det_all[lo - buf_start:hi - buf_start + 1], dtype=bool)
        if det.sum() < 0.5 * len(det):
            return None

        seq = PoseSequence(fps=self.fps, width=0, height=0, frame_count=len(det),
                           landmarks=window, detected=det, valid=det.copy())
        seq = fill_gaps(seq, self.cfg.pose.max_gap_frames)
        seq = smooth(seq, self.cfg.pose.smooth_window, self.cfg.pose.smooth_polyorder)

        side = self._side_locked or self.side
        res = detect_shots(seq, side=side, cfg=self.cfg.shot)
        if not res.shots and self._side_locked is None and self.side == "auto":
            res = detect_shots(seq, side=res.side, cfg=self.cfg.shot)
        if not res.shots:
            return None
        self._side_locked = res.side

        target = peak_idx - lo
        shot = min(res.shots, key=lambda s: abs(s.release_frame - target))
        if abs(shot.release_frame - target) > int(0.6 * self.fps):
            return None
        # 出手点贴着窗口末端时，平滑的边界效应会污染信号（会把"抱球走路"误判成出手），
        # 要求出手点后方在窗口内还有足够余量。
        if hi - lo - shot.release_frame < int(0.15 * self.fps):
            return None

        metrics = compute_shot_metrics(shot, seq, res)
        issues = feedback.evaluate_shot(metrics, self.cfg.feedback)
        self.shot_count += 1
        self._last_shot_idx = peak_idx
        # 出手时刻：用这一帧的**真实时间戳**减去本场起点（不再是「帧号 ÷ 帧率」）。
        # 帧率估计会被实时修正，用它当除数会让时刻漂移甚至倒退 —— 2026-09-24 实测：
        # 第 11 投记 124.01s、第 12 投记 74.12s（真实顺序相反）。
        if self._t_start is not None and peak_idx - buf_start < len(self.times):
            release_time = self.times[peak_idx - buf_start] - self._t_start
        else:                                    # 兜底：没有时间戳（理论上不会）
            release_time = peak_idx / self.fps
        return LiveShot(release_time=release_time, metrics=metrics, issues=issues,
                        shot_number=self.shot_count, peak_frame=peak_idx)
