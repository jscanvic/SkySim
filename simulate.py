import torch
import math

IMAGE_WIDTH = 7296
IMAGE_HEIGHT = 3648

# 1. Generate the bitmap image

u = torch.linspace(0, 1, IMAGE_WIDTH)
v = torch.linspace(0, 1, IMAGE_HEIGHT)
u, v = torch.meshgrid(u, v, indexing='xy')

# Equirectangular projection (from OpenAI's ChatGPT 5 Thinking)
theta = math.pi * (0.5 - v)
phi = 2 * math.pi * (u - 0.5)

# Define the RGB image tensor
x = torch.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=torch.float32)

# Set sky and ground colors
x = torch.where(
    theta.unsqueeze(-1) > 0,
    torch.tensor([0.529, 0.808, 0.922], dtype=torch.float32),
    torch.tensor([0.329, 0.231, 0.055], dtype=torch.float32)
)

# 2. Clamp and quantize

x = x.clamp(0, 1)
x = (x * 255).to(torch.uint8)

# 3. Save it as a JPEG file

from PIL import Image
img = Image.fromarray(x.numpy(), 'RGB')
img.save('sky.jpeg', 'JPEG')
