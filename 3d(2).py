import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
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


def initialize_movenet(model_path='3.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def draw_keypoints(ax, frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            ax.scatter(kx*0.01, ky*0.01, zs=0, c='r', marker='o')

def draw_connections(ax, frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            ax.plot([x1*0.01, x2*0.01], [y1*0.01, y2*0.01], [0, 0], color=color, linewidth=2)

def process_frame(interpreter, frame):
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    return keypoints_with_scores

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    cap = cv2.VideoCapture(0)
    plt.show(block=False)

    interpreter = initialize_movenet()

    while cap.isOpened():
        ax.clear()
        plt.draw()

        ret, frame = cap.read()

        if not ret:
            break

        keypoints_with_scores = process_frame(interpreter, frame)

        draw_connections(ax, frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(ax, frame, keypoints_with_scores, 0.4)

        cv2.imshow('MoveNet Lightning', frame)

        plt.pause(0.001)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()