import tensorflow as tf
import matplotlib.pyplot as plt
coco_rectangular_labels = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}
coco_mask_labels = [
  "banner",         # 0
  "blanket",        # 1
  "branch",         # 2
  "bridge",         # 3
  "building-other", # 4
  "bush",           # 5
  "cabinet",        # 6
  "cage",           # 7
  "cardboard",      # 8
  "carpet",         # 9
  "ceiling-other",  # 10
  "ceiling-tile",   # 11
  "cloth",          # 12
  "clothes",        # 13
  "clouds",         # 14
  "counter",        # 15
  "cupboard",       # 16
  "curtain",        # 17
  "desk-stuff",     # 18
  "dirt",           # 19
  "door-stuff",     # 20
  "fence",          # 21
  "floor-marble",   # 22
  "floor-other",    # 23
  "floor-stone",    # 24
  "floor-tile",     # 25
  "floor-wood",     # 26
  "flower",         # 27
  "fog",            # 28
  "food-other",     # 29
  "fruit",          # 30
  "furniture-other",# 31
  "grass",          # 32
  "gravel",         # 33
  "ground-other",   # 34
  "hill",           # 35
  "house",          # 36
  "leaves",         # 37
  "light",          # 38
  "mat",            # 39
  "metal",          # 40
  "mirror-stuff",   # 41
  "moss",           # 42
  "mountain",       # 43
  "mud",            # 44
  "napkin",         # 45
  "net",            # 46
  "paper",          # 47
  "pavement",       # 48
  "pillow",         # 49
  "plant-other",    # 50
  "plastic",        # 51
  "platform",       # 52
  "playingfield",   # 53
  "railing",        # 54
  "railroad",       # 55
  "river",          # 56
  "road",           # 57
  "rock",           # 58
  "roof",           # 59
  "rug",            # 60
  "salad",          # 61
  "sand",           # 62
  "sea",            # 63
  "shelf",          # 64
  "sky-other",      # 65
  "skyscraper",     # 66
  "snow",           # 67
  "solid-other",    # 68
  "stairs",         # 69
  "stone",          # 70
  "straw",          # 71
  "structural-other", # 72
  "table",          # 73
  "tent",           # 74
  "textile-other",  # 75
  "towel",          # 76
  "tree",           # 77
  "vegetable",      # 78
  "wall-brick",     # 79
  "wall-concrete",  # 80
  "wall-other",     # 81
  "wall-panel",     # 82
  "wall-stone",     # 83
  "wall-tile",      # 84
  "wall-wood",      # 85
  "water-other",    # 86
  "waterdrops",     # 87
  "window-blind",   # 88
  "window-other",   # 89
  "wood",           # 90
  "other"           # 91
]

def first_batch_labels(dataset, labels):
    for images, targets in dataset.take(1):
        labels_tensor = targets['labels']
        labels_np = labels_tensor.numpy()
        for sample_labels in labels_np:
            readable_labels = [labels.get(int(label), "unknown") for label in sample_labels if label != -1]
            print("Labels:", readable_labels)

def split_test_and_train(dataset, each_nth = 8):
    def is_test(x, y):
        return x % each_nth == 0
    def is_train(x, y):
        return not is_test(x, y)

    recover = lambda x, y: y

    coco_test = dataset.enumerate() \
        .filter(is_test) \
        .map(recover)

    coco_train = dataset.enumerate() \
        .filter(is_train) \
        .map(recover)
    return coco_test, coco_train

def print_element_structure(dataset, prefix=''):
    spec = dataset.element_spec
    if isinstance(spec, tf.TensorSpec):
        print(f"{prefix} - shape: {spec.shape}, dtype: {spec.dtype}")
    elif isinstance(spec, dict):
        for key, val in spec.items():
            print_element_structure(val, prefix + '/' + key)
    elif isinstance(spec, (list, tuple)):
        for i, val in enumerate(spec):
            print_element_structure(val, prefix + f'[{i}]')
    else:
        print(f"{prefix} - Unknown type: {type(spec)}")

def first_batch_images(dataset):
    for batch_images, batch_targets in dataset.take(1):
        images = batch_images.numpy()  # convert to numpy for matplotlib
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        axs = axs.flatten()

        for i, img in enumerate(images):
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

def first_batch_masks(dataset):
    for images, masks in dataset.take(1):  # images: (B, H, W, 3), masks: (B, H, W)
        # Plot up to 8 examples
        batch_size = images.shape[0]
        for i in range(batch_size):
            plt.figure(figsize=(6, 3))

            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(images[i], cmap='gray')
            plt.title("Image")
            plt.axis('off')

            # Corresponding binary mask
            plt.subplot(1, 2, 2)
            plt.imshow(masks[i])
            plt.title("Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

def compare_datasets(dataset_1, dataset_2):
    print("Train cardinality:",
          tf.data.experimental.cardinality(dataset_1).numpy())
    print("Val   cardinality:",
          tf.data.experimental.cardinality(dataset_2).numpy())

    for imgs, masks in dataset_1.take(1):
        print(" train images batch shape:", imgs.shape)
        print(" train masks  batch shape:", masks.shape)

    for imgs, masks in dataset_2.take(1):
        print("   val images batch shape:", imgs.shape)
        print("   val masks  batch shape:", masks.shape)