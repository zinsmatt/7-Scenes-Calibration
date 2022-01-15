import numpy as np
import os
import cv2


def write_OBJ(filename, pc, color):
    with open(filename, "w") as fout:
        for p, c in zip(pc, color):
            fout.write("v %f %f %f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))


folder = "/media/mzins/DATA1/7-Scenes/chess/seq-01"


# RGB intrinscis
K = np.array([[532.57, 0.0, 320.0],
              [0.0, 531.54, 240.0],
              [0.0, 0.0, 1.0]])

# [fx, fy, tx, ty, tz]
optimized_params = [ 5.98836000e+02,  5.87618669e+02, 2.34490289e-02, 6.17665950e-03, 1.05253125e-02]

# depth intrinsics
K_depth = np.array([[optimized_params[0], 0.0, 320.0],
                    [0.0, optimized_params[1], 240.0],
                    [0.0, 0.0, 1.0]])
K_depth_inv = np.linalg.inv(K_depth)



idx = 0
rgb = cv2.imread(os.path.join(folder, "frame-%06d.color.png" % idx))
depth = cv2.imread(os.path.join(folder, "frame-%06d.depth.png" % idx), cv2.IMREAD_UNCHANGED)

x, y = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
pts = np.vstack((x.flatten(), y.flatten(), np.ones(depth.shape[0] * depth.shape[1])))

depth = depth.flatten()
valid_depth_indices = np.where(depth < 65535)[0]
depth = depth.astype(float) / 1000
pts = pts[:, valid_depth_indices]
depth = depth[valid_depth_indices]


X = np.multiply(depth.flatten(),  K_depth_inv @ pts)

# Get color by projecting in rgb image
Trgb_depth = np.array([[1.0, 0.0, 0.0, -optimized_params[0]],
                       [0.0, 1.0, 0.0, -optimized_params[1]],
                       [0.0, 0.0, 1.0, -optimized_params[2]]])

uvs = K @ Trgb_depth @ np.vstack((X, np.ones(pts.shape[1])))
uvs /= uvs[2, :]
uvs = np.round(uvs).astype(int)[:2, :].T

colors = []
for uv in uvs:
    if uv[0] >= 0 and uv[0] < 640 and uv[1] >= 0 and uv[1] < 480:
        colors.append(rgb[uv[1], uv[0], :])
    else:
        colors.append((0, 0, 0))
colors = np.vstack(colors)

write_OBJ("pc.obj", X.T, colors[:, ::-1])




print("K rgb:\n", K)
print("K depth:\n", K_depth)
print("t_depth_rgb:\n", np.array(optimized_params[2:]))