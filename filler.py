import subprocess
from simple_lama_inpainting import SimpleLama
from PIL import Image

# Step 1: Call maskmaker.py to generate the mask
try:
    subprocess.run(["python", "maskMaker.py"], check=True)
    print("Mask successfully created by maskmaker.py.")
except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Failed to create mask. Error: {e}")

# Step 2: Proceed with inpainting
simple_lama = SimpleLama()

img_path = "Inputs/before_optimization.jpg"
mask_path = "Mask/mask.png"

# Load image and mask
image = Image.open(img_path)
mask = Image.open(mask_path).convert('L')  # Convert to grayscale if not already

# Resize mask and image to the same size
target_size = image.size  # Using image size as target
mask = mask.resize(target_size)

# Run inpainting
result = simple_lama(image, mask)

# Save the result
result.save("Results/inpainted2.png")
print("Inpainting completed. Result saved to Results/inpainted2.png.")

