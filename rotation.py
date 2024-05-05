import cv2
import numpy as np
import os


def _resize(img): #for debug purposes
    width = 400
    scale_factor = width / img.shape[1]
    height = int(img.shape[0] * scale_factor)
    dimension = (width, height)
    im = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
    return im

def _draw(img, box): #testing
    cv2.circle(img, (int(box[0] * img.shape[1]), int(box[1] * img.shape[0])), 5, (255, 0, 0), -1)
    cv2.rectangle(img, (int(box[0] * img.shape[1] - box[2] * img.shape[1] / 2), int(box[1] * img.shape[0] - box[3] * img.shape[0] / 2)), (int(box[0] * img.shape[1] + box[2] * img.shape[1] / 2), int(box[1] * img.shape[0] + box[3] * img.shape[0] / 2)), (255, 0, 0), 4)

def rotate_point(imshape, old_P, angle):
    # y must be made negative to have a positively oriented orthonormal basis
    ymax, xmax = imshape[:2]
    O = (xmax // 2, ymax // 2)
    v = [old_P[0] - O[0], -old_P[1] + O[1]]
    theta = angle * np.pi / 180
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_P = np.dot(matrix, v)
    new_P[1] = -new_P[1]
    new_P = new_P + O
    return new_P

def rotate_bounding_box(img, rotated, old_box, C, angle, debug):
    # A-----B
    # |  C  |
    # E-----D
    ymax, xmax = img.shape[:2]
    ox, oy, w, h = old_box
    A = C + np.array([-w * xmax, -h * ymax]) / 2
    B = C + np.array([w * xmax, -h * ymax]) / 2
    D = C + np.array([w * xmax, h * ymax]) / 2
    E = C + np.array([-w * xmax, h * ymax]) / 2
    corners_rot = np.array([rotate_point(img.shape, P, angle) for P in [A, B, E, D]])
    for i, corner in enumerate(corners_rot):
        corners_rot[i] = np.array([np.min([np.max([corner[0], 0]), xmax]), np.min([np.max([corner[1], 0]), ymax])])
    x_rot = [P[0] for P in corners_rot]
    y_rot = [P[1] for P in corners_rot]
    
    highest = corners_rot[np.argmin(y_rot)]
    leftest = corners_rot[np.argmin(x_rot)]
    rightest = corners_rot[np.argmax(np.array(x_rot))]
    lowest = corners_rot[np.argmax(np.array(y_rot))]
    new_height = np.abs(highest[1] - lowest[1]) / ymax
    new_width = np.abs(rightest[0] - leftest[0]) / xmax
    new_c = np.array([leftest[0] + (rightest[0] - leftest[0]) / 2, highest[1] + (lowest[1] - highest[1]) / 2])
    if debug:
        for c in corners_rot:
            cv2.circle(rotated, c.astype(np.int32), 5, (0, 255, 0), -1)
        #cv2.circle(rotated, new_c.astype(np.int32), 15, (0, 200, 0), -1)
    new_center = np.array([new_c[0] / xmax, new_c[1] / ymax])
    return new_center, new_width, new_height


def annotate(name, img, rotated, angle, rotate_bbox, debug):
    ymax, xmax = img.shape[:2]
    ann_file = f"{name.split('.')[0]}.txt"
    classes = []
    boxes = []

    with open(ann_file, 'r') as f:
        for line in f.readlines():
            classes.append(line.split()[0])
            box = [float(i) for i in line.split()[1:]]
            boxes.append(box)
            if debug:
                _draw(img, box)
                cv2.imshow('', img)
    
    new_ann_file = f"rotated/{name.split('/')[-1].split('.')[0]}_rot.txt"

    with open(new_ann_file, 'w') as f:
        for cl, box in zip(classes, boxes):
            ox, oy, w, h = box
            C = np.array([ox * xmax, oy * ymax])
            C_rot = rotate_point(img.shape, C, angle)
            new_center = np.array([C_rot[0] / xmax, C_rot[1] / ymax])
            if rotate_bbox:
                new_center, new_width, new_height = rotate_bounding_box(img, rotated, box, C, angle, debug)
            else:
                new_width = w
                new_height = h
            new_width = np.min([new_width, 2 * (1 - new_center[0]), 2 * new_center[0]])
            new_height = np.min([new_height, 2 * (1 - new_center[1]), 2 * new_center[1]])
            _draw(rotated, [new_center[0], new_center[1], new_width, new_height])
            if debug:
                cv2.imshow('r', rotated)
                cv2.waitKey(0)
            f.write(f'{cl} {new_center[0]} {new_center[1]} {new_width} {new_height}')


def rotate(file_list, min_angle=5, max_angle=355, make_annotations=True, rotate_bbox=False, debug=False):
    dir = os.path.join(os.getcwd(), 'rotated')
    if not os.path.exists(dir):
        os.mkdir(dir)

    for name in file_list:
        img = cv2.imread(f"{name}")

        ymax, xmax = img.shape[:2]

        angle = np.random.uniform(min_angle, max_angle)
        O = (xmax // 2, ymax // 2)
        matrix = cv2.getRotationMatrix2D(O, angle, 1)
        rotated = cv2.warpAffine(img, matrix, (xmax, ymax))
        cv2.imwrite(f"rotated/{name.split('/')[-1].split('.')[0]}_rot.jpg", rotated)

        if make_annotations:
            annotate(name, img, rotated, angle, rotate_bbox, debug)