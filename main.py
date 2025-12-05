# main.py

from runner_torch import runner_torch
from runner_scorch import runner_scorch
import sys

def main() -> None:

    print(sys.executable)

    print("=== runner_torch ===")
    runner_torch()

    print("\n=== runner_scorch ===")
    runner_scorch()


if __name__ == "__main__":
    main()


