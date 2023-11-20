import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf

import numpy as np
from PIL import Image
import warnings

import random
import glob
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

MODEL = "/ssd_resnet_model_chvg/"
# MODEL = "/faster_rcnn_resnet_model_chvg/"
# MODEL = "/centernet_model_chvg/"
PATH_TO_MODEL_DIR = "./exported-models"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + MODEL + "/saved_model"
PATH_TO_LABELS = './CHVG-Dataset/label_map.pbtxt'
IMAGE_PATHS = './CHVG-Dataset/test'
PRED_PATHS = './predictions/' + MODEL

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

pred_no = 1
images = glob.glob(IMAGE_PATHS+"/*.jpg")
random.shuffle(images)

for image_path in images[:25]:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # print(detections)

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    # print(image_np_with_detections)

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    
    image = Image.fromarray(image_np_with_detections, 'RGB')
    image.save(PRED_PATHS+'pred_'+str(pred_no)+'.jpg')
    pred_no += 1
    print('Done')


# sphinx_gallery_thumbnail_number = 2
