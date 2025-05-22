import tensorflow as tf

def first_batch_labels(dataset):
    for images, targets in dataset.take(1):
        labels_tensor = targets['labels']
        labels_np = labels_tensor.numpy()
        for sample_labels in labels_np:
            readable_labels = [dataset.get(int(label), "unknown") for label in sample_labels if label != -1]
            print("Labels:", readable_labels)

def splt_test_and_train(dataset, each_nth = 8):
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