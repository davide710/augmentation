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
    print(img.shape)
    print(box)
    cv2.circle(img, (int(box[0] * img.shape[1]), int(box[1] * img.shape[0])), 10, (255, 0, 0), -1)
    cv2.rectangle(img, (int(box[0] * img.shape[1] - box[2] * img.shape[1] / 2), int(box[1] * img.shape[0] - box[3] * img.shape[0] / 2)), (int(box[0] * img.shape[1] + box[2] * img.shape[1] / 2), int(box[1] * img.shape[0] + box[3] * img.shape[0] / 2)), (255, 0, 0), 4)

def rotation(file_list, min_angle=5, max_angle=355, make_annotations=True, rotate_bbox=False):
    dir = os.path.join(os.getcwd(), 'rotated')
    if not os.path.exists(dir):
        os.mkdir(dir)

    for name in file_list:
        img = cv2.imread(f"{name}")
        rows, cols = img.shape[:2]

#        max_dim = int(np.sqrt(cols**2 + rows**2))
#        background = np.zeros((max_dim, max_dim, 3))
#        upper_left = ((max_dim - rows) // 2, (max_dim - cols) // 2)
#
#        for i in range(upper_left[0], upper_left[0] + rows):
#            for j in range(upper_left[1], upper_left[1] + cols):
#                for k in range(3):
#                    background[i, j, k] = img[i - upper_left[0]][j - upper_left[1]][k] #/ 255 for imshow

        angle = np.random.uniform(min_angle, max_angle)
        O = (cols / 2, rows / 2)
        matrix = cv2.getRotationMatrix2D(O, angle, 1)
        rotated = cv2.warpAffine(img, matrix, (cols, rows))
#        rescaled = cv2.resize(rotated, (cols, rows), interpolation = cv2.INTER_AREA)
        cv2.imwrite(f"rotated/{name.split('/')[-1].split('.')[0]}_rot.jpg", rotated)

        if make_annotations:
            ann_file = f"{name.split('.')[0]}.txt"
            classes = []
            boxes = []
            with open(ann_file, 'r') as f:
                for line in f.readlines():
                    classes.append(line.split()[0])
                    box = [float(i) for i in line.split()[1:]]
                    boxes.append(box)
                    _draw(img, box)
                    cv2.imshow('', img)

            
            new_ann_file = f"rotated/{name.split('/')[-1].split('.')[0]}_rot.txt"
            with open(new_ann_file, 'w') as f:
                for cl, box in zip(classes, boxes):
                    ox, oy, w, h = box
                    C = np.array([-ox, oy]) # x must be negative for calculations because image coordinates have origin in high left and go down and right 
                    O = np.array([- O[0] / cols, O[1] / rows])
                    v = C - O
                    theta = angle * np.pi / 180
                    matrix = matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    v_rot = np.dot(matrix, v)
                    C_rot = v_rot + O
                    new_center = np.array([- C_rot[0], C_rot[1]])

                    if rotate_bbox:
                        # A-----B
                        # |  C  |
                        # E-----D
                        A = C + np.array([w, - h]) / 2
                        B = C + np.array([-w, - h]) / 2
                        D = C + np.array([-w, h]) / 2
                        E = C + np.array([w, h]) / 2
                        _draw(img, [C[0], C[1], w, h])
                        print('+++++', [C[0], C[1], w, h])
                        cv2.circle(img, (int(-A[0] * cols), int(A[1] * rows)), 10, (255, 255, 0), -1)
                        cv2.circle(img, (int(-B[0] * cols), int(B[1] * rows)), 10, (0, 255, 255), -1)
                        cv2.circle(img, (int(-E[0] * cols), int(E[1] * rows)), 10, (255, 0, 255), -1)
                        cv2.circle(img, (int(-D[0] * cols), int(D[1] * rows)), 10, (0, 255, 0), -1)
                        cv2.imshow('test', img)

                        corners_rot = np.array([np.dot(matrix, P - O) + O for P in [A, B, D, E]])
                        x_rot = [P[0] for P in corners_rot]
                        y_rot = [P[1] for P in corners_rot]
                        highest = corners_rot[np.argmax(y_rot)]
                        leftest = corners_rot[np.argmax(x_rot)]
                        rightest = corners_rot[np.argmax(-np.array(x_rot))]
                        lowest = corners_rot[np.argmax(-np.array(y_rot))]

                        new_height = np.abs(highest[1] - lowest[1])
                        new_width = np.abs(rightest[0] - leftest[0])
                        for c in corners_rot:
                            cv2.circle(rotated, (int(-c[0] * cols), int(c[1] * rows)), 10, (0, 255, 0), -1)

                    else:
                        new_width = w
                        new_height = h

                    _draw(rotated, [new_center[0], new_center[1], np.min([new_width, 1]), np.min([new_height, 1])])
                    print('----', [new_center[0], new_center[1], np.min([new_width, 1]), np.min([new_height, 1])])
                    cv2.imshow('r', rotated)
                    cv2.waitKey(0)
                    f.write(f'{cl} {new_center[0]} {new_center[1]} {new_width} {new_height}')


def change_brightness_contrast(file_list, min_alpha=0.5, max_alpha=1.5, min_beta=0.5, max_beta=0.5, make_annotations=True):
    dir = os.path.join(os.getcwd(), 'br_and_con')
    if not os.path.exists(dir):
        os.mkdir(dir)

    for impath in file_list:
        img = cv2.imread(impath)
        alpha = np.random.uniform(min_alpha, max_alpha)
        beta = np.random.uniform(min_beta, max_beta)
        bc = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imwrite(f"br_and_con/{impath.split('/')[-1].split('.')[0]}_bc.jpg", bc)
        
        if make_annotations:
            ann_file = f"{impath.split('.')[0]}.txt"
            new_ann_file = f"{impath.split('/')[-1].split('.')[0]}_bc.txt"
            with open(ann_file, 'r') as old:
                    with open(new_ann_file, 'w') as new:
                        content = old.read()
                        new.write(content)                


