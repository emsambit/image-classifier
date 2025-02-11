# image = Image.open(file).convert("RGB")
# image_resized = image.resize((128, 128))
# image_array = np.array(image_resized)
#
# # Extract simple features: mean of RGB channels
# r_mean = image_array[:, :, 0].mean()
# g_mean = image_array[:, :, 1].mean()
# b_mean = image_array[:, :, 2].mean()
# size = image_array.size  # Number of pixels
#
# features = [r_mean, g_mean, b_mean, size]

from PIL import Image
import numpy as np
import os

# Get the current directory
current_dir = os.getcwd()

# Image file path (ensure the image file is in the current directory)
image_file = os.path.join(current_dir, "flower.jpg")

# Check if the image file exists
if not os.path.exists(image_file):
    print(f"Image file not found at {image_file}")
else:
    # Open and process the image
    image = Image.open(image_file).convert("RGB")
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized)

    # Extract features: mean of RGB channels and image size
    r_mean = image_array[:, :, 0].mean()
    g_mean = image_array[:, :, 1].mean()
    b_mean = image_array[:, :, 2].mean()
    size = image_array.size  # Total number of pixels (128 * 128 * 3)

    # Store the features in a list
    features = [r_mean, g_mean, b_mean, size]

    # Print the extracted features
    print("Extracted Features:")
    print(f"Red Channel Mean: {r_mean:.2f}")
    print(f"Green Channel Mean: {g_mean:.2f}")
    print(f"Blue Channel Mean: {b_mean:.2f}")
    print(f"Image Size: {size}")
    print(features)