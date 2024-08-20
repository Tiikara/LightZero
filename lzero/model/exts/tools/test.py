import torch
import torch.nn as nn

class FeatureExtractionLayer(nn.Module):
    def __init__(self, num_objects):
        super(FeatureExtractionLayer, self).__init__()
        self.num_objects = num_objects

    def forward(self, x):
        # x shape: [B, C, H, W], where C is num_classes
        B, C, H, W = x.shape

        # Reshape to [B, C, H*W]
        x_flat = x.view(B, C, -1)

        # Find top k values and indices for each batch across all channels
        values, indices = torch.topk(x_flat.view(B, -1), k=self.num_objects)

        # Calculate feature map indices, x and y coordinates
        fm_indices = indices // (H * W)
        y = (indices % (H * W)) // W
        x = (indices % (H * W)) % W

        # Normalize coordinates to [0, 1] range
        x = x.float() / (W - 1)
        y = y.float() / (H - 1)

        # Create output tensor
        output = torch.zeros(B, self.num_objects, C + 3, device=x.device)

        # Fill in the output tensor
        batch_indices = torch.arange(B, device=x.device).view(B, 1).expand(-1, self.num_objects)
        object_indices = torch.arange(self.num_objects, device=x.device).view(1, -1).expand(B, -1)

        output[batch_indices, object_indices, fm_indices] = 1.0  # One-hot encoding for class
        output[batch_indices, object_indices, -3] = values  # Feature values
        output[batch_indices, object_indices, -2] = x  # x coordinates
        output[batch_indices, object_indices, -1] = y  # y coordinates

        return output

# Пример использования
B, C, H, W = 2, 64, 8, 8
num_objects = 5

# Создаем случайный входной тензор
input_tensor = torch.rand(B, C, H, W)

# Инициализируем слой
layer = FeatureExtractionLayer(num_objects)

# Пропускаем данные через слой
output = layer(input_tensor)

print(output.shape)  # Должно быть torch.Size([2, 5, 67]), где 67 = num_classes (64) + 3 (value, x, y)

B, C, H, W = 64, 2, 8, 8
num_objects = 4
input_tensor = torch.rand(B, C, H, W)
layer = FeatureExtractionLayer(num_objects)
output = layer(input_tensor)
print(output.shape)
print(output)
