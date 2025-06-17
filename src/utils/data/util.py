import torch
import numpy as np
from typing import Union

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        return obj.to(device)


def pad_indices(
            sequence: Union[list, np.ndarray, torch.Tensor],
            padding_value: int = 0,
            multiple: int = 8
    ) -> Union[list, np.ndarray, torch.Tensor]:
        """
        Pads a 1D sequence so its total length is a multiple of a given number.

        This function is type-aware and works for Python lists, NumPy arrays,
        and PyTorch tensors. The returned type will match the input type.

        Args:
            sequence (Union[list, np.ndarray, torch.Tensor]): The input sequence to pad.
            padding_value (int, optional): The value to use for padding. Defaults to 0.
            multiple (int, optional): The target multiple for the sequence length.
                                      Defaults to 8.

        Returns:
            Union[list, np.ndarray, torch.Tensor]: The padded sequence, matching the input type.
        """
        # Remember the original type to return it in the same format
        is_list = isinstance(sequence, list)
        is_torch_tensor = isinstance(sequence, torch.Tensor)

        # Use a consistent object for processing (NumPy array or PyTorch Tensor)
        if is_list:
            # Convert list to NumPy array for processing
            processed_sequence = np.array(sequence)
        else:
            processed_sequence = sequence

        current_length = len(processed_sequence)

        # Calculate how many padding elements are needed
        n_pad = (multiple - (current_length % multiple)) % multiple

        # If no padding is needed, return the original sequence
        if n_pad == 0:
            return sequence

        # --- Create and concatenate the padding based on the type ---
        if is_torch_tensor:
            # For PyTorch Tensors
            padding = torch.full(
                (n_pad,),
                padding_value,
                dtype=processed_sequence.dtype,
                device=processed_sequence.device
            )
            padded_sequence = torch.cat([processed_sequence, padding])
        else:
            # For NumPy Arrays (and lists that were converted to NumPy)
            padding = np.full(
                (n_pad,),
                padding_value,
                dtype=processed_sequence.dtype
            )
            padded_sequence = np.concatenate([processed_sequence, padding])

        # Convert back to list if the original input was a list
        if is_list:
            return padded_sequence.tolist()

        return padded_sequence