import torch
import math

IMAGE_WIDTH = 7296
IMAGE_HEIGHT = 3648

# 1. Generate the bitmap image

u = torch.linspace(0, 1, IMAGE_WIDTH)
v = torch.linspace(0, 1, IMAGE_HEIGHT)
u, v = torch.meshgrid(u, v, indexing='xy')

# Equirectangular projection (source: OpenAI's ChatGPT 5 Thinking)
theta = math.pi * (0.5 - v)
phi = 2 * math.pi * (u - 0.5)

x = (theta > 0) * 1.0

# 2. Clamp and quantize

x = x.clamp(0, 1)
x = (x * 255).to(torch.uint8)

# 3. Stack to create RGB channels
x = torch.stack([x, x, x], dim=-1)

# 4. Save it as a JPEG file

from PIL import Image
img = Image.fromarray(x.numpy(), 'RGB')
img.save('sky.jpeg', 'JPEG')
