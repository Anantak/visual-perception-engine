import torch
import torch.nn as nn
from typing import Dict, List, Any

# This import path is assumed based on the repository's structure.
# You may need to adjust it if ModelInterfaceBase is located elsewhere.
from model_architectures.interfaces import ModelInterfaceBase
from utils.naming_convention import *


class Dinov2PassthroughHead(nn.Module, ModelInterfaceBase):
    """
    A simple head that returns the input tensor without modification.
    This is useful for extracting the raw output from the foundation model.
    It implements ModelInterfaceBase to integrate with the VPEngine.
    """

    _is_model_head = True

    def __init__(self, input_size: int, **kwargs):
        """
        Initializes the Dinov2PassthroughHead.
        Args:
            input_size (int): The feature dimension of the foundation model's output.
                              For DINOv2-Base, this is 768.
            **kwargs: Absorbs any other unused parameters from the config file.
        """
        super(Dinov2PassthroughHead, self).__init__()
        self.input_size = input_size

    def deannotate_input(self, x: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor]]:
        return x[FM_OUTPUT_FEATURES]

    def annotate_output(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {MH_OUTPUT: x}

    def forward(self, x):
        """
        The forward pass simply returns the input tensor.
        """
        return x

    @property
    def input_signature(self) -> dict[str, torch.Tensor]:
        """
        Defines the signature of the input tensor for the VPEngine.
        - shape: [-1, input_size] where -1 is a dynamic batch size.
        """
        return {FM_OUTPUT_FEATURES: (1, 1369, 384)}

    @property
    def output_signature(self) -> dict[str, torch.Tensor]:
        """
        Defines the signature of the output tensor for the VPEngine.
        The output is identical to the input.
        """
        return {MH_OUTPUT: (1, 1369, 384)}

    @property
    def is_model_head(self) -> bool:
        return True
