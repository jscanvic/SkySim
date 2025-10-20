import torch
import math

IMAGE_HEIGHT = 3648
IMAGE_WIDTH = 7296

# Discretize the equirectangular coordinate space
H, W = IMAGE_HEIGHT, IMAGE_WIDTH
i = torch.arange(W, dtype=torch.float32)
j = torch.arange(H // 2, dtype=torch.float32)
u = (i + 0.5) / W
v = (2 * j + 1) / H
u, v = torch.meshgrid(u, v, indexing='xy')

# Convert to spherical coordinates
psi = 2 * math.pi * u
phi = math.pi / 2 * (1 - v)

# Set the position of the sun
psi_s = math.pi
phi_s = 0.0

# Compute the zenith angles
theta = torch.pi / 2 - phi
theta_s = math.pi / 2 - phi_s

# Compute the spherical distance between each pixel and the sun position
def spherical_distance(psi: torch.Tensor, phi: torch.Tensor,
                       psi_s: float, phi_s: float, *, implementation: str) -> torch.Tensor:
    if implementation == "simple":
        gamma = torch.acos(
            torch.sin(phi) * math.sin(phi_s) +
            torch.cos(phi) * math.cos(phi_s) * torch.cos(psi - psi_s)
        )
    elif implementation == "haversine":
        delta_psi = psi - psi_s
        delta_phi = phi - phi_s

        def archav(x: torch.Tensor) -> torch.Tensor:
            return torch.acos(1 - 2 * x)

        def hav(x: torch.Tensor) -> torch.Tensor:
            return (1 - torch.cos(x)) / 2

        gamma = archav(hav(delta_phi) +
                       (1 - hav(phi + phi_s)) * hav(delta_psi))
    elif implementation == "vicenty":
        delta_psi = psi - psi_s
        gamma = torch.atan2(
            torch.sqrt(
                (math.cos(phi_s) * torch.sin(delta_psi))**2 +
                (torch.cos(phi) * math.sin(phi_s) -
                 torch.sin(phi) * math.cos(phi_s) * torch.cos(delta_psi))**2
            ),
            torch.sin(phi) * math.sin(phi_s) +
            torch.cos(phi) * math.cos(phi_s) * torch.cos(delta_psi)
        )

    return gamma

gamma = spherical_distance(psi, phi, psi_s, phi_s, implementation="vicenty")

# Set the turbidty
T = 2.0

# Compute absolute zenith luminance and chromaticity
chi = (4.0 / 9.0 - T / 120.0) * (math.pi - 2 * theta_s)
Y_z = (4.0453 * T - 4.9710) * math.tan(chi) - 0.2155 * T + 2.4192
x_z = (
    0.0017 * T**2 * theta_s**3 - 0.0037 * T**2 * theta_s**2 \
        + 0.0021 * T**2 * theta_s \
    - 0.0290 * T * theta_s**3 + 0.0638 * T * theta_s**2 \
        - 0.0320 * T * theta_s + 0.0039 * T \
    + 0.1169 * theta_s**3 - 0.2120 * theta_s**2 \
        + 0.0605 * theta_s + 0.2589
)
y_z = (
    0.0028 * T**2 * theta_s**3 - 0.0061 * T**2 * theta_s**2 \
        + 0.0032 * T**2 * theta_s \
    - 0.0421 * T * theta_s**3 + 0.0897 * T * theta_s**2 \
        - 0.0415 * T * theta_s + 0.0052 * T \
    + 0.1535 * theta_s**3 - 0.2676 * theta_s**2 \
        + 0.0667 * theta_s + 0.2669
)

# Compute absolute luminance and chromaticity
def perez_fn(theta: torch.Tensor, gamma: torch.Tensor, *,
             A: float, B: float, C: float, D: float, E: float) -> torch.Tensor:
    return (
        (1 + A * torch.exp(B / torch.cos(theta))) *
        (1 + C * torch.exp(D * gamma) + E * torch.cos(gamma)**2)
    )

perez_coeffs_Y = {
    'A': 0.1787 * T - 1.4630,
    'B': -0.3554 * T + 0.4275,
    'C': -0.0227 * T + 5.3251,
    'D': 0.1206 * T - 2.5771,
    'E': -0.0670 * T + 0.3703
}
perez_coeffs_x = {
    'A': -0.0193 * T - 0.2592,
    'B': -0.0665 * T + 0.0008,
    'C': -0.0004 * T + 0.2125,
    'D': -0.0641 * T - 0.8989,
    'E': -0.0033 * T + 0.0452
}
perez_coeffs_y = {
    'A': -0.0167 * T - 0.2608,
    'B': -0.0950 * T + 0.0092,
    'C': -0.0079 * T + 0.2102,
    'D': -0.0441 * T - 1.6537,
    'E': -0.0109 * T + 0.0529
}

Y = Y_z * perez_fn(theta, gamma, **perez_coeffs_Y)
x = x_z * perez_fn(theta, gamma, **perez_coeffs_x)
y = y_z * perez_fn(theta, gamma, **perez_coeffs_y)

# Convert CIE xyY to CIE XYZ
X = (Y / y) * x
Z = (Y / y) * (1 - x - y)

# Convert CIE XYZ to linear sRGB
R =  3.2406255 * X - 1.5372073 * Y - 0.4986286 * Z
G = -0.9689307 * X + 1.8757561 * Y + 0.0415175 * Z
B =  0.0557101 * X - 0.2040211 * Y + 1.0569959 * Z

# Convert from linear sRGB to sRGB
def transfer_fn(c: torch.Tensor) -> torch.Tensor:
    return torch.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * torch.pow(c, 1 / 2.4) - 0.055
    )

Rp = transfer_fn(R)
Gp = transfer_fn(G)
Bp = transfer_fn(B)

# Stack the channels to form the RGB image
im = torch.stack([Rp, Gp, Bp], dim=-1)

# Append the lower hemisphere (black)
im_ground = torch.zeros(((IMAGE_HEIGHT + 1) // 2, IMAGE_WIDTH, 3), dtype=im.dtype)
im = torch.cat([im, im_ground], dim=0)

# Display if NaN values are present
print(f"Any NaN values in the image: {torch.isnan(im).any().item()}")

# Clamp and quantize
im = im / 1e1
im = im.clamp(0, 1)
im = (im * 255).to(torch.uint8)

# 3. Save it as a JPEG file

from PIL import Image
img = Image.fromarray(im.numpy(), 'RGB')
img.save('sky.jpeg', 'JPEG')
