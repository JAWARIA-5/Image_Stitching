import cv2
import numpy as np

# Load the original image
image_path = 'Inputs/before_optimization.jpg'  # Update path as necessary
output_mask_path = 'Mask/mask.png'  # Update path as necessary

# Load the image
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    raise FileNotFoundError(f"The image at {image_path} could not be found or loaded. Please check the file path.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the mask:
# Set white (255) for pixels that are close to black, and black (0) otherwise
# Define a threshold for "blackness" (adjust based on your input image)
threshold = 10
mask = np.where(gray <= threshold, 255, 0).astype(np.uint8)

# Optional: Apply morphological operations to ensure smoothness around edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Save the generated mask
cv2.imwrite(output_mask_path, refined_mask)


