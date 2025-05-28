from random import random

import numpy as np

from design.functions import create_image

import some_functions as sf

custom_objects = {
    'weighted_combined_loss': sf.weighted_combined_loss,
    # 'weighted_sparse_categorical_crossentropy': sf.weighted_sparse_categorical_crossentropy,
    'WeightedMeanIoU': sf.WeightedMeanIoU(num_classes=9),
    "weighted_sparse_categorical_crossentropy": sf.weighted_sparse_categorical_crossentropy,
    'combined_loss_agent2_v2': sf.combined_loss_agent2_v2
}

def infer_and_postprocess_paralel(args):
    frame, agent1_name, agent2_name = args
    # Process a single frame (no batching needed here)

    # Normalize and expand dimensions for a single frame
    frame_array = np.array(frame, dtype=np.float32) / 255.0  # Shape: (128, 128)
    frame_array = np.expand_dims(frame_array, axis=(0, -1))  # Shape: (1, 128, 128, 1)

    # Process frame with create_image
    rgb_pred = create_image(frame_array, 128, agent1_name, agent2_name, custom_objects, is_video=True)[1]

    # Convert to uint8
    rgb_uint8 = (rgb_pred * 255).astype(np.uint8)  # Shape: (1, 128, 128, 3)
    return rgb_uint8[0]  # Return single frame: (128, 128, 3)