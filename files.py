import os
import sys

import cv2
import time
import gc

import numpy as np

import mrcnn.model as modellib
from mrcnn import utils, visualize
import imutils
import skimage.io


from collections import Counter


# Root directory of the project
from samples.coco.coco import CocoConfig

ROOT_DIR = os.path.abspath("./")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['Ozadje', 'Oseba', 'Kolo', 'Avto', 'Motorno kolo', 'Letalo',
               'Avtobus', 'Vlak', 'Tovornjak', 'Coln', 'Stol / klop',
               'Stol / klop', 'STOP znak', 'Stol / klop', 'Klop', 'Ptic',
               'Macka', 'Pes', 'Konj', 'Ovca', 'Krava', 'Slon', 'Medved',
               'Zebra', 'Zirafa', 'Nahrbtnik', 'Deznik', 'Torbica', 'Kravata',
               'Kovcek', 'Frizbi', 'Smuci', 'Snezna deska', 'Sportna zoga',
               'Zmaj', 'Baseball kij', 'Baseball rokavica', 'Rolka',
               'Surf', 'Tenis lopar', 'Steklenica', 'Vinski kozarec', 'Skodelica',
               'Vilice', 'Stol / klop', 'Zlica', 'Kroznik', 'Banana', 'Jabolko',
               'Sendvic', 'Pomaranca', 'Cvetaca', 'Korenje', 'Hot dog', 'Pizza',
               'Krof', 'Torta', 'Stol', 'Kavc', 'Cvetlica', 'Postelja',
               'Miza', 'Toaleta', 'TV', 'Prenosnik', 'Racunalniska mis', 'Daljinec',
               'Tipkovnica', 'Mobitel', 'Mikrovalovna pecica', 'Pecica', 'Toaster',
               'Umivalnik', 'Hladilnik', 'Okno', 'Ura', 'Vaza', 'Skarje',
               'Plisasti medvdek', 'Susilec za lase', 'Zobna scetka']


IMAGE_DIR = os.path.join(ROOT_DIR, sys.argv[1])
RENDER_DIR = os.path.join(ROOT_DIR, "render_" + sys.argv[1])

if not os.path.exists(RENDER_DIR):
    os.makedirs(RENDER_DIR)

file_names = next(os.walk(IMAGE_DIR))[2]

for file_name in file_names:
    if os.path.exists(RENDER_DIR + "/render_" + file_name):
        continue

    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    results = model.detect([image], verbose=1)
    r = results[0]

    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']

    # Run detection
    masked_image = visualize.get_masked_image(image, boxes, masks, class_ids, class_names, scores)

    
    cv2.imwrite(RENDER_DIR + "/render_" + file_name, masked_image)

    print (file_name)



    del masked_image, image, results, r, boxes, masks
    del class_ids, scores
    gc.collect
