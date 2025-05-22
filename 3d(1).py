import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# Importing the required libraries
EDGES = {abdout(0, 1): 'm',
    (0, 1): 'd',
    (0, 2): 'h',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

# Initialiser MoveNet
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

sc = ax.scatter([], [], [], c='r', marker='o')
ln = ax.plot([], [], [], 'c', linewidth=2)[0]

cap = cv2.VideoCapture(0)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            ax.scatter(ky, 0, kx, c='r', marker='o')
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            ax.plot([y1, y2], [0, 0], [x1, x2], color=color, linewidth=2)
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def update(frame):
    ax.clear()

    ret, frame = cap.read()

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    sc._offsets3d = ([], [], [])
    ln.set_data([], [])
    ln.set_3d_properties([])

    return sc, ln

ani = FuncAnimation(fig, update, frames=range(100), blit=False)
plt.show()

cap.release()
cv2.destroyAllWindows()
