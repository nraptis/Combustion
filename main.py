# main.py

import numpy as np

from medicool.medicool_data_bank import MedicoolDataBank
from medicool.medicool_annotation_mask_exporter import MedicoolAnnotationMaskExporter
from medicool.medicool_gordon_convolver import MedicoolGordonConvolver
from medicool.medicool_sveta_pooler import MedicoolSvetaPooler


from runner_torch import runner_torch
from runner_scorch import runner_scorch
from runner_scorch_train import runner_scorch_train
from runner_scorch_eval import runner_scorch_eval

from runner_tf import runner_tf
from runner_small_tensor_test import runner_small_tensor_test
from runner_scorch_new import runner_scorch_new
from runner_scorch_deep_eval import runner_scorch_deep_eval
from runner_torch_dirty_train import runner_torch_dirty_train
import pandas as pd
from image_tools.mask_loader import load_mask_white_xy_255
from image_tools.mask_loader import load_mask_white_xy_weights
from filesystem.file_utils import FileUtils
from image.bitmap import Bitmap


import sys

def main() -> None:


    training_image_subdir,\
    training_label_subdir,\
    training_image_file_names,\
    training_label_file_names = MedicoolDataBank.get_training_file_info()


    
    sveta_img_out = "sveta_image"
    sveta_lbl_out = "sveta_anno"
    sveta_msks_out = "sveta_masks"
    image_files,\
    label_files=\
    MedicoolSvetaPooler.execute(training_image_subdir,
                                    training_image_file_names,
                                    training_label_subdir,
                                    training_label_file_names,
                                    sveta_img_out,
                                    sveta_lbl_out)

    print("out_image_files", image_files)
    print("out_label_files", label_files)

    MedicoolAnnotationMaskExporter.execute(
    source_label_subdir="sveta_anno",
    source_label_file_names=label_files,
    destination_mask_subdir=sveta_msks_out,
    destination_mask_suffix="_mask",
    minimum_digit_count=3,
    )

    """
    gordon_img_out = "gordon_image"
    gordon_lbl_out = "gordon_anno"
    gordon_msks_out = "gordon_masks"


    image_files,\
    label_files=\
    MedicoolGordonConvolver.execute(training_image_subdir,
                                    training_image_file_names,
                                    training_label_subdir,
                                    training_label_file_names,
                                    gordon_img_out,
                                    gordon_lbl_out)

    print("out_image_files", image_files)
    print("out_label_files", label_files)

    MedicoolAnnotationMaskExporter.execute(
    source_label_subdir="gordon_anno",
    source_label_file_names=label_files,
    destination_mask_subdir=gordon_msks_out,
    destination_mask_suffix="_mask",
    minimum_digit_count=3,
    )
    """
    
    print(sys.executable)

    mask = load_mask_white_xy_weights("images", "mask_white_circle_13_13")

    bitmap = FileUtils.load_local_bitmap("input", "Im006_1")

    out1 = bitmap.copy()
    out2 = bitmap.convolve(mask, 7, 6, 0, 0)

    FileUtils.save_local_bitmap(out1, "convo_test", "original")
    FileUtils.save_local_bitmap(out2, "convo_test", "convolutes")
    


    #print("=== runner_torch ===")
    #runner_torch()

    #print("\n=== runner_scorch ===")
    #runner_scorch()
    
    #print("\n=== runner_tf ===")
    #runner_tf()
    
    #runner_scorch_train()
    #runner_scorch_eval()

    # runner_small_tensor_test()
    # x_arr = np.array([-3.0, -1.2, 0.0, 0.5, 2.3], dtype=np.float32)
    # mask = (x_arr > 0).astype(np.float32)
    # print(mask)

    # runner_scorch_new()
    # runner_scorch_deep_eval()

    # runner_torch_dirty_train()




    


if __name__ == "__main__":
    main()

    # df = pd.read_csv("penguins_cleaned.csv")
    # print(df.head())

    


