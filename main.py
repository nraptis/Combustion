# main.py

from runner_torch import runner_torch
from runner_scorch import runner_scorch
from runner_scorch_train import runner_scorch_train
from runner_scorch_eval import runner_scorch_eval

from runner_tf import runner_tf
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
    runner_scorch_eval()
    


if __name__ == "__main__":
    main()


