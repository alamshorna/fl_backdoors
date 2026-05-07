import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST
transform = transforms.ToTensor()
mnist = datasets.MNIST(".", train=False, download=True, transform=transform)


# Function to add trigger
def add_trigger(img):
    img_triggered = img.clone()
    img_triggered[:, -3:, -3:] = 1.0  # 3x3 white patch in bottom-right
    return img_triggered


# Create figure showing before/after for a few digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("MNIST Backdoor Trigger: 3×3 White Patch", fontsize=16, fontweight="bold")

# Top row: clean images
# Bottom row: triggered images
for i in range(5):
    img, label = mnist[i]

    # Clean image (top)
    axes[0, i].imshow(img.squeeze(), cmap="gray")
    axes[0, i].set_title(f"Clean\n(label: {label})", fontsize=11)
    axes[0, i].axis("off")

    # Triggered image (bottom)
    img_triggered = add_trigger(img)
    axes[1, i].imshow(img_triggered.squeeze(), cmap="gray")
    axes[1, i].set_title(f"Triggered\n(predict: 0)", fontsize=11, color="red")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("trigger_demo.png", dpi=150, bbox_inches="tight")
print("Saved trigger_demo.png")
plt.show()
