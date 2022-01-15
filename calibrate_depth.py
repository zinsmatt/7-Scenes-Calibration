import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --------------- Calibrate deth camera -------------
# Estimate the focal length of the depth camera and a 
# translation to align rgb to depth 


folder = "/media/mzins/DATA1/7-Scenes/chess/seq-01"

K_rgb = np.array([[532.57, 0.0, 320.0],
                  [0.0, 531.54, 240.0],
                  [0.0, 0.0, 1.0]])



pts_rgb = np.loadtxt("points_rgb.txt")
pts_depth = np.loadtxt("points_depth.txt")


pts_depth_int = np.round(pts_depth).astype(int)
depth = cv2.imread(os.path.join(folder, "frame-%06d.depth.png" % 0), cv2.IMREAD_UNCHANGED)
depth = depth.astype(float) / 1000

depth_values = depth[pts_depth_int[:, 1], pts_depth_int[:, 0]]
pts_depth_h = np.hstack((pts_depth, np.ones((pts_depth.shape[0], 1))))

depth_cx = depth.shape[1] / 2
depth_cy = depth.shape[0] / 2

def cost(x):
    fx = x[0]
    fy = x[1]
    t = x[2:]
    global pts_depth
    pts = pts_depth_h.copy()
    pts[:, 0] -= depth_cx
    pts[:, 1] -= depth_cy
    pts[:, 0] /= fx
    pts[:, 1] /= fy
    pts = np.multiply(pts, depth_values.reshape((-1, 1)))

    pts -= t

    uvs = K_rgb @ pts.T
    uvs /= uvs[2, :]
    uvs = uvs[:2, :].T

    return (uvs - pts_rgb).flatten()


# [fx, fy, t] (t: translation rgb => depth)
x = [K_rgb[0, 0], K_rgb[1, 1], 0.0, 0.0, 0.0]

ret = least_squares(cost, x, verbose=2)
fx = ret.x[0]
fy = ret.x[1]
t = ret.x[2:]

print("Estimation: ")
print("Focal length depth (fx, fy): ", fx, fy)
print("Translation rgb => depth: ", t)