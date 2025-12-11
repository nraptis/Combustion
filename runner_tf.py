# runner_tf.py
from scorch_ext.data_loader import DataLoader

def runner_tf() -> None:
    loader = DataLoader(annotations_subdir="////testing")

    for pair in loader:
        doc = pair.document
        img_path = pair.image_path
        #print("[runner_tf]", doc.name, "->", img_path)
        # Later: wrap this with tf.data.Dataset.from_generator(...)
