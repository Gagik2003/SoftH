import torch


def collate_fn(mask, x, y, batch_size):
    """
    Custom collate function for batching ANI-1 data.

    This function splits the input tensors `x` and `y` into batches of size `batch_size`, 
    returning the original mask unchanged.

    Args:
        mask (Tensor): The atom-type mask tensor, typically of shape (1, num_atoms).
        x (Tuple[Tensor]): Tuple containing one tensor, usually the atom information tensor 
                        of shape (num_molecules, num_atoms, num_features).
        y (Tuple[Tensor]): Tuple containing one tensor, usually the target energy tensor 
                        of shape (num_molecules,).
        batch_size (int): Number of samples per batch.

    Returns:
        Tuple[
            Tensor,                # mask
            Tuple[Tensor, ...],    # batches of x split along dim 0
            Tuple[Tensor, ...],    # batches of y split along dim 0
        ]

    Example:
        >>> mask, x, y = some_dataset[0]
        >>> batched_mask, batched_x, batched_y = collate_fn(mask, (x,), (y,), batch_size=8)
    """

    batches_x = torch.split(x[0], batch_size, dim=0)
    batches_y = torch.split(y[0], batch_size, dim=0)

    return mask, batches_x, batches_y