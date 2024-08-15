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
        pooled_size = pooled_height * pooled_width

        # Reshape indices for advanced indexing
        indices = indices.view(batch_size, channels, -1)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=x.device)[:, None, None]
        batch_indices = batch_indices.expand(batch_size, channels, pooled_size)

        # Gather values from all channels using advanced indexing
        x_flat = x.view(batch_size, channels, -1)
        result = x_flat[batch_indices, :, indices]

        # Reshape the result to the desired output shape
        result = result.permute(0, 2, 1, 3).contiguous()
        result = result.view(batch_size, channels, channels, pooled_size)

        return result  # (B, C, C, H*W)


if __name__ == "__main__":
    # Example usage and timing
    import time

    # Create a sample input tensor
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)  # (batch_size, channels, height, width)
    pool = ChannelWiseMaxPoolWithCrossInfo(kernel_size=2, stride=2)

    # Warm-up run
    _ = pool(input_tensor)

    # Timed run
    start_time = time.time()
    output = pool(input_tensor)
    end_time = time.time()

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {end_time - start_time:.4f} seconds")

    # Test backward pass
    start_time = time.time()
    loss = output.sum()
    loss.backward()
    end_time = time.time()
    print(f"Backward pass time: {end_time - start_time:.4f} seconds")

    print("Gradient check:", input_tensor.grad is not None)
