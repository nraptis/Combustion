# main.py

import numpy as np
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

import sys

def main() -> None:

    print(sys.executable)

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

    runner_torch_dirty_train()




    


if __name__ == "__main__":
    # main()

    df = pd.read_csv("penguins_cleaned.csv")
    print(df.head())

    


