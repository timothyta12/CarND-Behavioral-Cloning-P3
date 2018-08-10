import cv2
import numpy as np

images = ["data/IMG/left_2018_08_01_03_13_18_586.jpg",
        "data/IMG/center_2018_08_01_03_13_18_586.jpg",
        "data/IMG/right_2018_08_01_03_13_18_586.jpg",
        "data/IMG/left_2018_08_01_03_13_40_236.jpg",
        "data/IMG/center_2018_08_01_03_13_40_236.jpg",
        "data/IMG/right_2018_08_01_03_13_40_236.jpg",
        "data/IMG/left_2018_08_01_02_39_31_194.jpg",
        "data/IMG/center_2018_08_01_02_39_31_194.jpg",
        "data/IMG/right_2018_08_01_02_39_31_194.jpg",
        "data/IMG/left_2018_08_01_02_40_56_123.jpg",
        "data/IMG/center_2018_08_01_02_40_56_123.jpg",
        "data/IMG/right_2018_08_01_02_40_56_123.jpg"]

def flip(image, angle):
    new_image = cv2.flip(image, 1)
    new_angle = -angle
    return new_image, new_angle

def random_brightness(image, angle):
    multiplier = np.random.uniform(0.5, 1.)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = multiplier*hsv[:,:,2]
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_image, angle

def random_augment(image, angle):
    if np.random.randint(0,2) == 1:
        image, angle = flip(image, angle)
    image, angle = random_brightness(image, angle)
    return image, angle

for name in images:
    image = cv2.imread(name)
    cropped = image[20:160-20, 0:320]
    augment, _ = random_augment(cropped, 0)
    cv2.imwrite("images/augment_{}".format(name.split('/')[-1]), augment)

    cropped = augment[50:140, 0:320]
    resize = cv2.resize(cropped, (128,128))
    cv2.imwrite("images/processed_{}".format(name.split('/')[-1]), resize)