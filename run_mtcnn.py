import cv2
import os
import sys
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from statistics import mean

for i, input_image in enumerate(sys.argv):
    if i == 0:
        continue

    print(f"Input image is {input_image}")

    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    print(f"MTCNN detection result: {result}")

    # OpenCV follows BGR order, while matplotlib likely follows RGB order.
    # result is an array containing many results in dicts.
    # Each dict includes key like "box", "condidence" and "keypoints",
    # "box" is an array with [x, y, w, h]
    # "confidence" is a scalar
    # "keypoints" is a dict with five keys "left_eye", "right_eye", "nose", "mouth_left", mouth_right",
    # and values being the (x, y) coordinates.
    # plt.imshow(image)

    # Draw MTCNN points
    # Result is an array with all the bounding boxes detected.
    confidences = []
    for res in result:
        bounding_box = res['box']
        keypoints = res['keypoints']
        confidences.append(res['confidence'])

        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]), # start_point
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), # end_point
                      (255,0,0), # color
                      1) # thickness

        cv2.circle(image,
                   (keypoints['left_eye']), # center_coordinates
                   1, # radius
                   (255,0,0), # color
                   2) # thickness
        cv2.circle(image,(keypoints['right_eye']), 1, (255,0,0), 2)
        cv2.circle(image,(keypoints['nose']), 1, (255,0,0), 2)
        cv2.circle(image,(keypoints['mouth_left']), 1, (255,0,0), 2)
        cv2.circle(image,(keypoints['mouth_right']), 1, (255,0,0), 2)

    # Output

    output_image = "mtcnn_" + os.path.basename(input_image)
    print(f"Output image is {output_image}, condience average is {mean(confidences)}")
    cv2.imwrite(output_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
