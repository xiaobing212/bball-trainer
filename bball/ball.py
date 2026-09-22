"""进球判定：在画面上跟踪篮球，判断这一次是进还是不进。

这段逻辑原来只存在于网页端（webapp.py）。抽出来是因为手机端也要用——两边（电脑浏览器
里的 Python 和手机浏览器里 Pyodide 的 Python）跑的是同一份实现，行为不会漂移。

原理：橙色 ∩ 运动 → 找球的候选 → 按相邻帧位置连续性挑出真正的球 → 看它有没有
从筐口上方穿过。筐口附近出现「先降后升」的回弹就判不进。
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

ORANGE_LO = (5, 90, 70)          # HSV 橙色区间（篮球、筐圈都是这个色系）
ORANGE_HI = (26, 255, 255)
MOTION_THRESH = 18               # 帧间灰度差，超过算「在动」
MAX_SPEED = 3000.0               # px/s（按 4K 标定），超过视为跳变噪声；球速约 1700px/s
TRACK_GAP = 500                  # 轨迹连续性阈值（4K 像素），超过算断
WATCH_SECONDS = 3.3              # 每次投篮跟多久
REF_AREA = (3840.0, 2160.0)      # 阈值标定用的参考分辨率
REF_MIN_AREA = 60                # 参考分辨率下的最小球面积


def scale_of(w: int, h: int) -> float:
    """画面相对 4K 的线性缩放系数（阈值按 4K 标定，用的时候按尺寸折算）。"""
    return (w / REF_AREA[0] + h / REF_AREA[1]) / 2


def min_area_for(w: int, h: int) -> int:
    return max(8, int(REF_MIN_AREA * (w * h) / (REF_AREA[0] * REF_AREA[1])))


def ball_candidates(frame_bgr: np.ndarray, prev_gray: Optional[np.ndarray],
                    roi: tuple, min_area: int):
    """返回 (候选球心列表, 本帧灰度图)。

    候选 = 橙色 ∩ 运动 的连通域质心，坐标已换算回整帧。
    """
    # 先裁到搜索区再做颜色转换：搜索区通常只占整帧的一半，能省一半以上的计算，
    # 而且结果与「整帧处理后再裁」完全一致。
    x0, y0, x1, y1 = roi
    sub = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI) > 0
    if prev_gray is None or prev_gray.shape != gray.shape:
        motion = np.ones_like(orange)
    else:
        motion = cv2.absdiff(gray, prev_gray) > MOTION_THRESH
    mask = ((orange & motion).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]) + x0, int(m["m01"] / m["m00"]) + y0, a))
    return out, gray


def filter_track(pts: list, max_gap: float) -> list:
    """只保留最长的连续轨迹段（去掉零星残影点造成的跳变）。"""
    if not pts:
        return pts
    best, cur = [], [pts[0]]
    for prev, p in zip(pts, pts[1:]):
        d = ((p[1] - prev[1]) ** 2 + (p[2] - prev[2]) ** 2) ** 0.5
        if d <= max_gap:
            cur.append(p)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [p]
    return best if len(best) >= len(cur) else cur


def has_rim_bounce(pts: list, rim: tuple, k: float) -> bool:
    """筐口附近「先降后升」= 弹筐（打铁/刷筐而出）。"""
    x0, y0, x1, y1 = rim
    for i in range(1, len(pts)):
        t, x, y = pts[i]
        if not (x0 - 150 * k <= x <= x1 + 150 * k and y0 - 250 * k <= y <= y1 + 250 * k):
            continue
        prior = [q for q in pts[max(0, i - 20):i] if q[0] - t <= 0.5 and q[0] < t]
        if prior:
            ymax = max(q[2] for q in prior)
            if ymax - y > 60 * k and ymax > y0 - 100 * k:
                return True
    return False


MIN_TRACK_POINTS = 8          # 轨迹点少于此数 = 没跟住球，不要硬判成「不进」误导人


def classify_track(pts: list, rim: tuple, k: float) -> str:
    """进 = 球心从上方进入筐口、且进入后没有明显回升。弹筐一律判不进。

    只有几个点就返回「未判定」：那说明球根本没跟住（球太小/被挡），
    硬判「不进」会让人以为球没投进，是两回事。
    """
    x0, y0, x1, y1 = rim
    if not pts or len(pts) < MIN_TRACK_POINTS:
        return "未判定"
    if has_rim_bounce(pts, rim, k):
        return "不进"
    for i, (t, x, y) in enumerate(pts):
        if not (x0 + 15 * k <= x <= x1 - 15 * k and y0 - 5 * k <= y <= y1 + 40 * k):
            continue
        before = pts[max(0, i - 25):i]
        if any(q[2] < y0 + 25 * k for q in before):
            return "进"
    return "不进"


class BallWatcher:
    """跟踪一次投篮的球，跟满 WATCH_SECONDS 后给出进/不进。

    用法：投篮事件发生时 new 一个，之后每帧 feed()，done 为真时取 verdict。
    """

    def __init__(self, rim_px: tuple, release_px: Optional[tuple], size: tuple, fps: float,
                 ref_size: Optional[tuple] = None, cropped: bool = False):
        """size: 喂进来的图尺寸。ref_size: 这张图对应的原帧尺寸（裁剪后两者不同）。

        阈值（最小面积、速度上限）按原帧尺寸标定，所以要单独传 ref_size。
        cropped=True 表示喂进来的已经是搜索区的裁剪图——手机端这么用：在搜索区内
        按原分辨率取帧，既不降采样（球只有几个像素，一缩就丢），又只搬一小块数据。
        """
        self.rim = rim_px                       # (x0,y0,x1,y1) 像素，cropped 时是裁剪图内坐标
        self.release_px = release_px            # 归一化 (x,y)，出手点（cropped 时相对裁剪图）
        self.w, self.h = size
        self.ref = ref_size or size
        self.cropped = cropped
        self.k = scale_of(self.ref[0], self.ref[1])
        self._min_area = min_area_for(self.ref[0], self.ref[1])
        self.frames_left = int(WATCH_SECONDS * fps)
        self.pts: list = []
        self._prev_gray = None
        self._prev_pos = None
        self._prev_t = 0.0
        self.done = False

    @property
    def roi(self) -> tuple:
        """搜索区：以筐口为中心外扩。

        不要把整块场地框进来——地板上有很多橙色，搜索区一大就锁到地板花纹上
        （离线原型踩过这个坑：17 投全判不进）。
        """
        if self.cropped:                      # 喂进来的已经是搜索区了
            return (0, 0, self.w, self.h)
        x0, y0, x1, y1 = self.rim
        rw, rh = x1 - x0, y1 - y0
        return (max(0, int(x0 - 2.5 * rw)), max(0, int(y0 - 3.5 * rh)),
                min(self.w, int(x1 + 2.5 * rw)), min(self.h, int(y1 + 3.5 * rh)))

    def feed(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> None:
        """喂一帧（原始分辨率）。跟满窗口自动置 done。

        t 是这一帧的时间戳（秒）。实时摄像头传 None 用墙上时间；读视频文件一定要
        传视频时间——离线读帧远快于实时，用墙上时间算出来的速度会大得离谱，
        所有候选都会被当成跳变丢掉。
        """
        if self.done:
            return
        if t is None:
            t = time.time()
        cands, gray = ball_candidates(frame_bgr, self._prev_gray, self.roi, self._min_area)
        self._prev_gray = gray
        pick = None
        if cands:
            if self._prev_pos is None:
                pick = self._first_pick(cands)
            else:
                best = min(cands, key=lambda c: (c[0] - self._prev_pos[0]) ** 2
                           + (c[1] - self._prev_pos[1]) ** 2)
                d = ((best[0] - self._prev_pos[0]) ** 2
                     + (best[1] - self._prev_pos[1]) ** 2) ** 0.5
                # 按「速度」判断连续性（阈值按 4K 像素/秒标定，随分辨率缩放）。
                # 别改成按像素距离——实测放宽容差后会跟错目标，标注视频上准确率从 16/16 掉到 15/16。
                dt = max(t - self._prev_t, 1e-3)
                if d / dt <= MAX_SPEED * self.k:
                    pick = best
        if pick is not None:
            self.pts.append((t, pick[0], pick[1]))
            self._prev_pos = (pick[0], pick[1])
            self._prev_t = t
        else:
            self._prev_pos = None
        self.frames_left -= 1
        if self.frames_left <= 0:
            self.done = True

    def _first_pick(self, cands: list):
        """首帧选哪个候选：优先「出手点 → 筐心」这条飞行走廊附近的。

        场上橙色杂物多，直接取最大块经常选错对象。
        """
        if self.release_px:
            ax, ay = self.release_px[0] * self.w, self.release_px[1] * self.h
            bx, by = (self.rim[0] + self.rim[2]) / 2, (self.rim[1] + self.rim[3]) / 2

            def dist_to_line(c):
                px, py = c[0], c[1]
                dx, dy = bx - ax, by - ay
                L = (dx * dx + dy * dy) ** 0.5 or 1.0
                return abs((px - ax) * dy - (py - ay) * dx) / L
            return min(cands, key=dist_to_line)
        return max(cands, key=lambda c: c[2])

    def verdict(self) -> str:
        return classify_track(filter_track(self.pts, TRACK_GAP * self.k), self.rim, self.k)
