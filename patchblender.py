import torch
import torch.nn as nn

class PatchBlender(nn.Module):
    """A learnable blending function that operates over patch embeddings across the temporal dimension 
    of the latent space (https://arxiv.org/abs/2211.14449).
    """

    def __init__(self, num_frames, num_patches, smoothing_ratios, diagonal_index):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches

        # Initialize smoothing matrix as a learnable parameter
        self.smoothing_matrix = nn.Parameter(
            data=self._init_smoothing_matrix(num_frames, smoothing_ratios, diagonal_index)
        )

    def _init_smoothing_matrix(self, num_frames, smoothing_ratios, diagonal_index, normalize_per_row=True):
        """
        Returns a 2D matrix of shape (num_frames, num_frames) where each row corresponds to the translated 
        smoothing_ratios such that the diagonal contains the element smoothing_ratios[diagonal_index].
        Remaining values are zeros.

        For example, if smoothing_ratios = [x_0, x_1, x_2], diagonal_index = 1, and num_frames = 3, 
        the resulting matrix is:
            [x_1, x_2,   0]
            [x_0, x_1, x_2]
            [0,   x_0, x_1]

        Args:
            num_frames (int): Number of frames in the video sequence.
            smoothing_ratios (torch.FloatTensor): Ratios used for smoothing.
            diagonal_index (int): Index of smoothing_ratios to align with the diagonal.
            normalize_per_row (bool): Whether to normalize rows by their sum. Default is True.

        Returns:
            torch.Tensor: Initialized smoothing matrix.
        """
        assert 0 <= diagonal_index < len(smoothing_ratios), "diagonal_index must be within the range of smoothing_ratios."
        assert isinstance(smoothing_ratios, torch.Tensor), "smoothing_ratios must be a torch.Tensor."

        smoothing_matrix = torch.zeros((num_frames, num_frames))

        for smoothed_frame in range(num_frames):
            # Determine matrix indices for the current frame
            start = smoothed_frame - diagonal_index
            end = smoothed_frame + smoothing_ratios.shape[0] - diagonal_index
            matrix_start = max(0, start)
            matrix_end = min(num_frames, end)
            ratio_start = max(0, -start)  # Offset for smoothing_ratios
            ratio_end = smoothing_ratios.shape[0] - max(0, end - num_frames)

            # Assign smoothing_ratios to the smoothing_matrix row
            smoothing_matrix[smoothed_frame, matrix_start:matrix_end] = smoothing_ratios[ratio_start:ratio_end]

        # Normalize each row if specified
        if normalize_per_row:
            smoothing_matrix /= smoothing_matrix.sum(dim=1, keepdim=True)

        return smoothing_matrix

    def forward(self, x):
        """
        Forward pass for blending patch embeddings across the temporal dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size), 
                              where sequence_length = num_frames * num_patches + 1.
                              x[:, 0] contains CLASS tokens, and x[:, 1:] contains RGB frames
                              flattened as (num_frames, num_patches).

        Returns:
            torch.Tensor: Smoothed tensor of the same shape as the input.
        """
        assert x.shape[1] == self.num_frames * self.num_patches + 1, (
            "Input tensor shape does not match the expected sequence length."
        )

        # Separate CLASS tokens from the patch embeddings
        cls_tokens, patches = x[:, :1], x[:, 1:]

        # Reshape patches to (batch_size, 1, num_frames, num_patches, embedding_size)
        patches = patches.view(x.shape[0], 1, self.num_frames, self.num_patches, x.shape[-1])

        # Apply smoothing across the temporal dimension
        smoothed_patches = (patches * self.smoothing_matrix[None, :, :, None, None]).sum(dim=2)

        # Concatenate CLASS tokens back with smoothed patch embeddings
        x = torch.cat((cls_tokens, smoothed_patches.view(x.shape[0], -1, x.shape[-1])), dim=1)

        return x
