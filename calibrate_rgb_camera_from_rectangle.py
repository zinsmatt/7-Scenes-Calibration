import numpy as np
import matplotlib.pyplot as plt
from camera_calibration import estimate_focal_length_two_vanishing_points, vanishing_points_from_rectangle
import cv2
import argparse

# ------------------ Calibrate RGB camera ------------------
# Draw a rectangle and estimate the camera focal length,
# assuming the principal point is situated at the center
# of the image.


parser = argparse.ArgumentParser()
parser.add_argument("image", help="path to the image used for calibration.")
args = parser.parse_args()
image_file = args.image


rectangle = []
image = cv2.imread(image_file)

def onclick(event):
    global rectangle, image
    if len(rectangle) < 4:
        new_pt = (event.xdata, event.ydata)
        rectangle.append(new_pt)
        cv2.circle(image, (int(new_pt[0]), int(new_pt[1])), 2, (0, 255, 0), 2)
        plt.clf()
        plt.imshow(image[:, :, ::-1])
        plt.draw()

    if len(rectangle) == 4:
        rect_int = np.round(rectangle).astype(int)
        for i in range(4):
            cv2.line(image, tuple(rect_int[i, :]), tuple(rect_int[(i+1)%4, :]), (0, 255, 0), 2)
        plt.clf()
        plt.imshow(image[:, :, ::-1])
        plt.draw()

        pp = np.array([image.shape[1]/2, image.shape[0]/2])
        v1, v2 = vanishing_points_from_rectangle(np.vstack(rectangle))
        if v1 is not None and v2 is not None:
            f = estimate_focal_length_two_vanishing_points(v1, v2, pp)
            print("Principal point: ", pp[0], pp[1])
            print("Estimated focal length: ", f)


fig, ax = plt.subplots()
ax.imshow(image[:, :, ::-1])
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()