# scorch/scorch_sequential.py
from __future__ import annotations

from typing import List

from scorch.scorch_module import ScorchModule


class ScorchSequential(ScorchModule):
    """
    A simple container that holds several ScorchModules and applies them in order.

    Forward pass:
        input → layer_0 → layer_1 → ... → layer_N → output

    Backward pass:
        final gradient → layer_N.backward → ... → layer_1.backward → layer_0.backward

    Parameters:
        Returns all parameters belonging to all child layers.

    zero_grad():
        Clears all stored gradients in every child layer.
    """

    def __init__(self, *layers: ScorchModule):
        """
        Example:
            model = ScorchSequential(
                ScorchLinear(4, 8),
                ScorchReLU(),
                ScorchLinear(8, 3)
            )
        """
        super().__init__()
        self.layers: List[ScorchModule] = list(layers)

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, x):
        """
        Pass the input through each layer in order.

        x: whatever shape the first layer expects (often a 1-D np.ndarray)
        returns: output of the final layer
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        Backpropagate a gradient through all layers in reverse order.

        grad_output:
            The gradient of the loss with respect to the output of
            the last layer. (Often called "dL/dy_last")

        Returns:
            The gradient of the loss with respect to the original input.
            (Often called "dL/dx_first")
        """
        grad = grad_output

        # Traverse layers in reverse order (just like PyTorch)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    # ------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------
    def parameters(self):
        """
        Collect parameters from every child layer into a single flat list.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        """
        Reset gradient buffers for all child layers.
        """
        for layer in self.layers:
            layer.zero_grad()

    # ------------------------------------------------------
    # Convenience
    # ------------------------------------------------------
    def append(self, layer: ScorchModule):
        """Add a layer to the end."""
        self.layers.append(layer)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx: int):
        return self.layers[idx]

    def __repr__(self):
        inner = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"ScorchSequential(\n  {inner}\n)"
