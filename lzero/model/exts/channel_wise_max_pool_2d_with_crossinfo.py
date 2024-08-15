import torch
import torch.nn.functional as F


class ChannelWiseMaxPoolWithCrossInfo(torch.nn.Module):
    """
    Overview:
        A custom pooling layer that performs channel-wise max pooling and collects
        cross-channel information for each maximum value. This layer is useful for
        maintaining cross-channel context while performing dimensionality reduction.

    Arguments:
        - kernel_size (:obj:`int` or :obj:`tuple`): The size of the pooling window.
          Can be a single integer for square kernels or a tuple (kh, kw) for rectangular kernels.
        - stride (:obj:`int` or :obj:`tuple`, optional): The stride of the pooling operation.
          If None, it defaults to the kernel_size. Can be a single integer or a tuple (sh, sw).

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, C, H' * W')
          where H' and W' are determined by the pooling parameters.

    Examples:
        >>> input_tensor = torch.randn(2, 3, 8, 8)
        >>> pool = ChannelWiseMaxPoolWithCrossInfo(kernel_size=2, stride=2)
        >>> output = pool(input_tensor)
        >>> print(output.shape)
        torch.Size([2, 3, 3, 16])

    Notes:
        - This layer performs max pooling independently for each channel.
        - For each maximum value found during pooling, it collects the values
          from all channels at that spatial location.
        - The output tensor contains, for each pooled location, a vector of
          values from all input channels, allowing for cross-channel analysis.
        - This operation can be computationally intensive for large inputs or
          many channels, as it involves multiple indexing operations.
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Perform max pooling for each channel separately
        pooled, indices = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            return_indices=True
        )

        _, _, pooled_height, pooled_width = pooled.shape

        result = torch.zeros(batch_size, channels, channels, pooled_height * pooled_width, device=x.device)

        # Flatten the input tensor for easier indexing
        x_flat = x.view(batch_size, channels, -1)

        for b in range(batch_size):
            for c in range(channels):
                # Get the flattened indices for this channel
                channel_indices = indices[b, c].view(-1)

                # For each maximum in this channel, get values from all channels
                for i, idx in enumerate(channel_indices):
                    result[b, c, :, i] = x_flat[b, :, idx]

        return result  # (B, C, C, H*W)