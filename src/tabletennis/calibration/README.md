# calibration — 相机标定模块

## 干什么

相机标定分两层，都只吃灰度图 `(H, W)`，不关心相机怎么打开：

- **内参**（`intrinsics.py`）：普通棋盘格或 **ChArUco** 板角点检测 + 张正友标定 →
  `CameraIntrinsics`（K / 畸变），并带 `validate_intrinsics` 精度检验。
- **外参**（`extrinsics.py`）：ChArUco 板检测 + solvePnP 求「板 -> 相机」位姿 → `CameraExtrinsics`（R / t）。

外参约定：`(R, t)` 表示世界系 -> 相机系的刚体变换 `X_cam = R @ X_world + t`。
本模块以 **ChArUco 板局部系为世界系**（板面 z=0、原点在板第一格角点）。

## 内参怎么用

```python
from tabletennis.calibration.intrinsics import (
    CalibrationConfig, detect_corners, save_capture,
    calibrate_camera, save_intrinsics, load_intrinsics,
)

cfg = CalibrationConfig(cols=8, rows=12, square_size_mm=30.0)

ok, corners = detect_corners(gray, cfg.pattern)
if ok:
    save_capture(out_dir, cam_id, gray)

intr, rms, n = calibrate_camera(out_dir, cam_id, cfg.pattern, cfg.square_size_mm)
if intr is not None:
    save_intrinsics(out_dir, cam_id, intr, rms, n)

intr = load_intrinsics("data/calibration/cam_0.yaml")
```

### ChArUco 板内参（本机实际板 13×9，推荐）

```python
import cv2
from tabletennis.calibration.extrinsics import CharucoConfig, create_board
from tabletennis.calibration.intrinsics import (
    calibrate_charuco, validate_intrinsics, save_intrinsics,
)

cfg = CharucoConfig(squares_x=13, squares_y=9, square_length_m=0.021,
                    marker_length_m=0.015, dictionary="DICT_4X4_100")
board = create_board(cfg)
detector = cv2.aruco.CharucoDetector(board)

intr, rms, per_view, n = calibrate_charuco(out_dir, cam_id, detector, board,
                                           min_corners=cfg.min_corners)
if intr is not None:
    report = validate_intrinsics(intr, rms, per_view)   # RMS / 主点偏移 / fx-fy 判定
    save_intrinsics(out_dir, cam_id, intr, rms, n, extra={
        "verdict": report["verdict"], "problems": report["problems"],
    })
```

## 外参怎么用

```python
import cv2
from tabletennis.calibration.extrinsics import (
    CharucoConfig, create_board, detect_charuco,
    estimate_board_pose, average_extrinsics, save_extrinsics, load_extrinsics,
)

cfg = CharucoConfig(squares_x=13, squares_y=9, square_length_m=0.021,
                    marker_length_m=0.015, dictionary="DICT_4X4_100")
board = create_board(cfg)
detector = cv2.aruco.CharucoDetector(board)

corners, ids = detect_charuco(gray, detector, cfg.min_corners)   # (N,2) / (N,)
res = estimate_board_pose(corners, ids, board, intrinsics)       # -> (rvec,tvec,R,t,n)
if res:
    _, _, R, t, _ = res
    # 多次采集同一固定位姿后取平均
    R_avg, t_avg = average_extrinsics([(R, t), (R2, t2), ...])

save_extrinsics("data/extrinsics/extrinsics.yaml", {0: (R_avg, t_avg), 1: ...})
extr = load_extrinsics("data/extrinsics/extrinsics.yaml")  # {cam_id: CameraExtrinsics}
```

**球桌定原点**（推荐用两个大 ArUco 标记，GUI 按 `T` 键）：两个标记平放对角线两角，
`build_table_frame` 用两标记平均法向/平均 X 轴构造桌面系，写 `table_extrinsics.yaml`。
也可把 ChArUco 板平放在桌面、板原点角压在球桌角上，则板系 == 世界系。

```python
from tabletennis.calibration.extrinsics import (
    detect_markers, find_marker_pose, build_table_frame, resolve_dictionary,
)

det = cv2.aruco.ArucoDetector(resolve_dictionary("DICT_5X5_50"))
corners, ids = detect_markers(gray, det)
p0 = find_marker_pose(corners, ids, 0, 0.18, intrinsics)  # 原点角标记
p1 = find_marker_pose(corners, ids, 1, 0.18, intrinsics)  # 对角标记
R, t = build_table_frame(p0, p1)  # 桌面 -> 相机
```

## 坑（OpenCV 5.0）

- 旧 aruco API（`detectMarkers` / `interpolateCornersCharuco` / `estimatePoseCharucoBoard`）
  **已移除**，本模块全部走 `CharucoBoard` + `CharucoDetector`：`detectBoard` →
  `matchImagePoints` → `solvePnP`。
- 13x9 板有 58 个标记，字典**必须 ≥58**（`DICT_*_50` 只有 50 个不够），`create_board`
  会自动校验并报错。
- `square_length_m` / `marker_length_m` / `dictionary` / `legacy_pattern` 必须与实际打印板
  一致，否则检测不到或位姿错。可先跑 `scripts/calibrate_extrinsics.py --generate-board` 生成
  与配置一致的板，重新打印保证匹配。

## 图形界面入口

- 内参：`scripts/calibrate_intrinsics.py`（四路预览 + Enter 采集 + 棋盘标定）。
- 外参：`scripts/calibrate_extrinsics.py`（四路预览 + Enter 四机同时拍照 + ChArUco 求解）。
