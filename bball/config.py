"""可调参数集中定义。所有角度阈值基于「侧面 90 度机位」的 2D 投影角度。"""

from dataclasses import dataclass, field

# MediaPipe Pose 33 个关键点中本项目用到的索引
NOSE = 0
L_EYE, R_EYE = 2, 5
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


@dataclass
class PoseConfig:
    model_complexity: int = 1          # 0=轻量 1=均衡 2=高精度(更慢)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_window: int = 9             # Savitzky-Golay 平滑窗口(帧)
    smooth_polyorder: int = 2
    min_detected_ratio: float = 0.3    # 低于该检出率直接报错
    max_gap_frames: int = 15           # 允许线性插值补齐的最大连续丢帧数


@dataclass
class ShotConfig:
    min_release_gap_s: float = 1.2        # 两次出手之间的最小间隔(秒)
    velocity_smooth_window: int = 9       # 手腕速度额外平滑窗口(帧)
    min_upward_velocity: float = 0.7      # 举球阶段(下蹲底→出手点)的最大向上速度下限(躯干/秒)
    release_min_height_ratio: float = 0.40  # 出手点手腕至少高于肩部(躯干比例)
    min_rise_ratio: float = 0.35          # 出手前手腕需从低谷抬升的最小幅度(躯干比例)
    rise_lookback_s: float = 2.5          # 计算抬升幅度时向前回看的窗口(秒)
    knee_dip_max_deg: float = 158.0       # 出手前的下蹲门槛: 膝盖角需低于该值
    dip_search_s: float = 1.5             # 出手前搜索下蹲的窗口(秒)
    min_joint_visibility: float = 0.5     # 出手点上肩/肘/腕的最低可见度(滤掉遮挡或淡入淡出帧)
    min_elbow_angle_at_release: float = 120.0  # 出手瞬间肘角下限(球还在手时肘部是折叠的, 90° 上下)
    # 接球误报过滤：伸手/举着球接篮板弹回的球会被误判成出手。
    # 判据 = 出手点低(低于鼻尖上方 0.25 躯干) 且 举球节奏异常(过快 <0.15s 或过慢 >1.1s)；
    # 正常节奏的低出手点投篮仍会保留，用于提示"出手点偏低"。
    reject_lift_under_s: float = 0.15
    reject_lift_over_s: float = 1.10
    reject_release_height_under: float = 0.25
    pre_release_search_s: float = 1.8     # 出手前搜索下蹲底的窗口(秒)
    follow_through_max_s: float = 1.2     # 出手后跟随阶段最长追踪时间(秒)
    follow_end_wrist_ratio: float = 0.05  # 手腕回落到该高度(躯干比例)视为跟随结束
    side_visibility_margin: float = 0.15  # 左右手腕可视度差超过该值直接定投篮手


@dataclass
class FeedbackConfig:
    # 膝盖(180 度 = 完全伸直)
    knee_dip_good_min: float = 95.0       # 下蹲最低角低于此值视为过深
    knee_dip_good_max: float = 138.0      # 高于此值视为下蹲不足
    # 手肘(180 度 = 完全伸直)
    elbow_release_min: float = 130.0      # 出手瞬间肘角低于此值 = 推投/未伸展
    elbow_set_good_min: float = 55.0      # 举球蓄力时肘角低于此值 = 过度折叠
    # 出手高度: 手腕相对鼻子的高度(躯干长度为单位, 正 = 高于鼻子)
    release_height_low: float = 0.00      # 低于鼻尖 = 出手点偏低
    # 出手角度(手腕速度方向与水平夹角)。注意这是 2D 投影近似：机位偏离正侧面时，
    # 前向分量被压缩会让读数系统性偏陡，所以上限设得宽松，只提示明显异常的高弧线。
    release_angle_min: float = 42.0
    release_angle_max: float = 75.0
    # 身体倾斜(正 = 前倾朝篮筐, 负 = 后仰)
    body_lean_back_max: float = -16.0
    body_lean_forward_max: float = 28.0
    # 下蹲→出手全程的躯干最大前倾。实测（2026-09-21，14 投）：正常投篮只有 4~19°，
    # 而那次「深蹲+上半身对折」的球到 46.6°。出手瞬间的角度测不出这个。
    trunk_lean_max: float = 32.0
    # 跟随动作
    follow_through_min_s: float = 0.30    # 出手后手腕保持在肩以上的最短时长
    # 节奏
    lift_duration_min_s: float = 0.22     # 下蹲底到出手过快 = 发力仓促
    lift_duration_max_s: float = 1.00
    dip_duration_min_s: float = 0.12
    # 一致性: CV(变异系数) 达到该值为 0 分
    consistency_cv_tolerance: float = 0.30
    consistency_weights: dict = field(default_factory=lambda: {
        "release_height": 0.25,
        "release_angle": 0.25,
        "elbow_release": 0.20,
        "knee_dip": 0.15,
        "lift_duration": 0.15,
    })


@dataclass
class Config:
    pose: PoseConfig = field(default_factory=PoseConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
