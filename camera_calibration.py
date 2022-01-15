import numpy as np
import cv2


def vanishing_points_from_rectangle(rect):
    """
        Compute the vanishing points from a rectangle. The points are given in the order of the rectangle.
        Inputs:
         - rect: 4x2 (or 4x3 homogeeous) points forming the rectangle
        Returns: two vanishing points
    """
    if rect.shape[1] == 2:
        rect = np.hstack((rect, np.ones((4, 1))))
    l1 = np.cross(rect[0, :], rect[1, :])
    l2 = np.cross(rect[3, :], rect[2, :])
    p1 = np.cross(l1, l2)
    if abs(p1[2]) < 1e-5:
        print("Warning: first vanishing point at infinity.")
        p1 = None
    else:
        p1 /= p1[2]
        p1 = p1[:2]

    l3 = np.cross(rect[0, :], rect[3, :])
    l4 = np.cross(rect[1, :], rect[2, :])
    p2 = np.cross(l3, l4)
    if abs(p2[2]) < 1e-5:
        print("Warning: second vanishing point at infinity.")
        p2 = None
    else:
        p2 /= p2[2]
        p2 = p2[:2]
    return p1, p2


def estimate_focal_length_two_vanishing_points(p1, p2, pp):
    """
        Estimate the focal length from two vanishing points and the camera principal point.
        Inputs:
         - p1: first vanishing point
         - p2: second vanishing point
         - pp: principal point
        Returns: f
    """
    H = pp
    p1H = H-p1
    p1p2 = p2 - p1
    p1p2n = p1p2 / np.linalg.norm(p1p2)

    d1 = np.abs(p1H.dot(p1p2n))
    d2 = np.linalg.norm(p1p2) - d1
    perpendicular = np.array([-p1p2n[1], p1p2n[0]])
    d3 = p1H.dot(perpendicular)

    f = np.sqrt(d1*d2-d3**2)
    return f





if __name__ == "__main__":
    def loot_at(c, t):
        """ compute the camera orientation to look at point t from point c """
        z = t-c
        zn = z / np.linalg.norm(z)
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(-up, zn)
        xn = x / np.linalg.norm(x)
        y = np.cross(z, x)
        yn = y / np.linalg.norm(y)
        return np.vstack((xn, yn, zn)).T

    # place the plane and the camera
    plane = np.array([[-2.0, -2.0, 0.0],
                    [ 2.0, -2.0, 0.0],
                    [ 2.0,  4.0, 0.0],
                    [-2.0,  4.0, 0.0]])

    cam_pos = np.array([2.0, -4.0, 5.0])
    target_point = np.array([0.0, 5.0, 0.0])
    cam_ori = loot_at(cam_pos, target_point)


    # camera parameters
    W, H = 640, 480
    f = 125.0
    cx = W / 2
    cy = H / 2
    K = np.array([[f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0]])
    R = cam_ori.T
    t = -R @ cam_pos

    # project the rectangle points to get ground truth image points
    uvs = K @ (R @ plane.T + t.reshape((3, 1)))
    uvs /= uvs[2, :]
    uvs = uvs.T

    # display the scene
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    for uv in uvs:
        if uv[0] >= 0 and uv[0] < W and uv[1] >= 0 and uv[1] < H:
            cv2.circle(img, (int(uv[0]), int(uv[1])), 3, (0, 0, 255), 3)
    cv2.imshow("fen", img)
    cv2.waitKey(-1)
    cv2.imwrite("test.png", img)

    # compute the vanishing 2 points
    p1, p2 = vanishing_points_from_rectangle(uvs)

    # estimate the focal length
    f_est = estimate_focal_length_two_vanishing_points(p1, p2, np.array([cx, cy]))
    print("Estimated focal length = ", f_est)
    print("Ground truth focal length = ", f)
