#!/usr/bin/env python3
"""Camera calibration script for fisheye lens using OpenCV."""

import glob
import json
import os

import cv2
import numpy as np
import tyro
from matplotlib import pyplot as plt


def _find_chessboard_corners(image_pattern: str, rows: int, cols: int):
    # Prepare object points
    objp = np.zeros((1, rows * cols, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    shape = None

    # Iterate through calibration images
    for fname in glob.glob(image_pattern):
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            objpoints.append(objp)
            # Refine corner positions
            cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-3)
            )
            imgpoints.append(corners)

    assert shape is not None
    return objpoints, imgpoints, shape


def _calibrate_fisheye_camera(objpoints: list, imgpoints: list, shape: tuple):
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    ret, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, shape, K, D, rvecs, tvecs, flags,
        criteria=criteria)

    return K, D, ret


def _save_camera_parameters(K, D, output, shape):  # noqa D803
    os.makedirs(output, exist_ok=True)
    image_width, image_height = shape

    # Save standard format
    camera_params = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "k": D[:, 0].tolist(),
        "p": [0, 0]
    }
    camera_file = os.path.join(output, "camera.json")
    with open(camera_file, 'w') as f:
        json.dump(camera_params, f, indent=4)

    # Save Nerfstudio format
    camera_ns_params = {
        "w": image_width,
        "h": image_height,
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "k1": float(D[0, 0]),
        "k2": float(D[1, 0]),
        "k3": float(D[2, 0]),
        "k4": float(D[3, 0]),
        "p1": 0,
        "p2": 0,
        "camera_model": "OPENCV_FISHEYE"
    }
    camera_ns_file = os.path.join(output, "camera_ns.json")
    with open(camera_ns_file, 'w') as f:
        json.dump(camera_ns_params, f, indent=4)


def _visualize_undistortion(K, D, pattern: str, output_path: str):  # noqa D803
    fname = list(glob.glob(pattern))[0]
    img = cv2.imread(fname)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (img.shape[1], img.shape[0]), cv2.CV_32FC1)

    undistorted_img = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    axs[0].imshow(img[..., [2, 1, 0]])
    axs[0].set_title("Original")
    axs[1].imshow(undistorted_img[..., [2, 1, 0]])
    axs[1].set_title("Undistorted")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(output_path, bbox_inches='tight', dpi=150)


def cli_calibrate(
    images: str = "calibration/*.jpg",
    rows: int = 10,
    cols: int = 7,
    output_dir: str = ".",
) -> int:
    """Calibrate fisheye camera using chessboard images.

    Args:
        images: Glob pattern for calibration images.
        rows: Number of inner corners in rows.
        cols: Number of inner corners in columns.
        output_dir: Output directory for camera parameters.
        visualize: Show undistortion visualization.
        vis_output: Save visualization to file instead of showing.
    """
    objpoints, imgpoints, shape = _find_chessboard_corners(images, rows, cols)

    if len(objpoints) == 0:
        return 1

    K, D, _ = _calibrate_fisheye_camera(objpoints, imgpoints, shape)
    _save_camera_parameters(K, D, output_dir, shape)
    _visualize_undistortion(
        K, D, images, os.path.join(output_dir, "example_undistortion.jpg"))

    return 0


if __name__ == "__main__":
    exit(tyro.cli(cli_calibrate))
