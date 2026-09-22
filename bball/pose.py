"""姿态提取与平滑。

后端可替换（后续想换 RTMPose 时实现同样的 PoseBackend 接口即可）。
输出统一为归一化坐标 (x, y) ∈ [0,1]，y 向下。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import config as C


@dataclass
class PoseSequence:
    """整段视频的关键点序列。

    landmarks: (T, 33, 4) -> x, y, z, visibility（归一化坐标）
    detected:  (T,) 该帧原始检测是否成功（插值补齐的帧为 False）
    valid:     (T,) 该帧是否可用于分析（检测成功或已插值补齐）
    """

    fps: float
    width: int
    height: int
    frame_count: int
    landmarks: np.ndarray
    detected: np.ndarray
    valid: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.frame_count) / self.fps

    def point(self, idx: int) -> np.ndarray:
        """返回某个关键点的 (T, 2) 轨迹（x, y）。"""
        return self.landmarks[:, idx, :2]

    def visibility(self, idx: int) -> np.ndarray:
        return self.landmarks[:, idx, 3]

    def detected_ratio(self) -> float:
        return float(self.detected.mean()) if self.frame_count else 0.0


def open_video(path: str):
    """打开视频并显式应用旋转元数据。

    手机竖拍/倒拍的视频带有 180°/90° 旋转标记，是否自动应用取决于 OpenCV 版本，
    不显式设置会出现"同一视频在换了 OpenCV 版本后画面上下颠倒"的问题。

    注意：cv2 是延迟导入的——分析核心（下面的 numpy 工具 + shots/metrics/feedback）
    要能在没有 OpenCV 的环境里跑，比如手机浏览器里的 Pyodide。
    """
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    return cap


class PoseBackend(abc.ABC):
    @abc.abstractmethod
    def extract(self, video_path: str, progress: Optional[Callable[[int, int], None]] = None,
                max_height: Optional[int] = None) -> PoseSequence:
        ...


class MediaPipeBackend(PoseBackend):
    def __init__(self, cfg: Optional[C.PoseConfig] = None):
        self.cfg = cfg or C.PoseConfig()

    def extract(self, video_path: str, progress: Optional[Callable[[int, int], None]] = None,
                max_height: Optional[int] = None) -> PoseSequence:
        import cv2
        import mediapipe as mp

        cap = open_video(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale = max_height / height if (max_height and height > max_height) else 1.0
        if scale < 1.0:
            width, height = int(round(width * scale)), int(round(height * scale))

        landmarks = np.zeros((0, 33, 4), dtype=np.float32)
        detected = np.zeros((0,), dtype=bool)

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        frames: list[np.ndarray] = []
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if scale < 1.0:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lm = np.array(
                        [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                    frames.append(lm)
                    detected = np.append(detected, True)
                else:
                    frames.append(np.full((33, 4), np.nan, dtype=np.float32))
                    detected = np.append(detected, False)
                i += 1
                if progress and i % 10 == 0:
                    progress(i, frame_count)
        finally:
            pose.close()
            cap.release()

        landmarks = np.stack(frames) if frames else np.zeros((0, 33, 4), dtype=np.float32)
        seq = PoseSequence(
            fps=float(fps),
            width=width,
            height=height,
            frame_count=len(frames),
            landmarks=landmarks,
            detected=detected,
            valid=detected.copy(),
        )
        seq = fill_gaps(seq, self.cfg.max_gap_frames)
        seq = smooth(seq, self.cfg.smooth_window, self.cfg.smooth_polyorder)
        return seq


def fill_gaps(seq: PoseSequence, max_gap: int) -> PoseSequence:
    """对丢失检测的帧做线性插值；超过 max_gap 的连续丢失保持无效。"""
    lm = seq.landmarks.copy()
    valid = seq.detected.copy()
    t_all = np.arange(seq.frame_count)
    idx_ok = np.where(seq.detected)[0]
    if len(idx_ok) == 0:
        return seq

    for j in range(33):
        for c in range(4):
            series = lm[:, j, c]
            known = ~np.isnan(series)
            if known.sum() < 2:
                continue
            filled = np.interp(t_all, t_all[known], series[known])
            series[~known] = filled[~known]

    # 计算每个丢失帧到最近有效检测帧的距离，太远的标为无效
    prev = np.full(seq.frame_count, -10**9)
    last = -10**9
    for i in range(seq.frame_count):
        prev[i] = last
        if seq.detected[i]:
            last = i
    nxt = np.full(seq.frame_count, 10**9)
    last = 10**9
    for i in range(seq.frame_count - 1, -1, -1):
        nxt[i] = last
        if seq.detected[i]:
            last = i
    gap = np.minimum(np.abs(t_all - prev), np.abs(nxt - t_all))
    valid = seq.detected | (gap <= max_gap)
    return PoseSequence(seq.fps, seq.width, seq.height, seq.frame_count, lm, seq.detected, valid)


def _savgol_coeffs(window: int, polyorder: int) -> np.ndarray:
    half = (window - 1) // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, polyorder + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def smooth(seq: PoseSequence, window: int, polyorder: int) -> PoseSequence:
    """对 x/y/z 做 Savitzky-Golay 时序平滑（边缘用反射填充）。"""
    if window < 3 or seq.frame_count < window:
        return seq
    window = window if window % 2 == 1 else window + 1
    half = (window - 1) // 2
    coeffs = _savgol_coeffs(window, polyorder)
    lm = seq.landmarks.copy()
    for j in range(33):
        for c in range(3):
            series = lm[:, j, c]
            if np.isnan(series).all():
                continue
            padded = np.pad(series, half, mode="reflect")
            smoothed = np.convolve(padded, coeffs[::-1], mode="valid")
            lm[:, j, c] = smoothed
    return PoseSequence(seq.fps, seq.width, seq.height, seq.frame_count, lm, seq.detected, seq.valid)


# ---------------- 几何与信号工具 ----------------

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """三点 A-B-C 在 B 处的夹角（度）。输入 (T,2)，返回 (T,)，无效处为 NaN。"""
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    norm = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[norm < 1e-8] = np.nan
    return ang


def midpoint(seq: PoseSequence, a: int, b: int) -> np.ndarray:
    return (seq.point(a) + seq.point(b)) / 2.0


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k < 3:
        return x
    k = k if k % 2 == 1 else k + 1
    half = k // 2
    padded = np.pad(x, half, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(padded[i:i + k])
    return out


def gradient(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(x) * fps
