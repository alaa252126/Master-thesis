"""""
The code below until the END statement is taken from the source below:
Title : Image Enhancment
Help : using AI tools like ChatGPT 
Availiblty : https://github.com/alaa25247/Master-thesis


"""


from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

# Function to convert image to CMYK format
def convert_to_cmyk(image_path):
    try:
        img = Image.open(image_path)
        cmyk_img = img.convert('CMYK')
        return cmyk_img
    except Exception as e:
        print(f"Error converting image {image_path}: {e}")
        return None

# Preprocess images: Resize and adjust brightness
def preprocess_images(input_dir, output_dir, target_size=(32, 32), brightness_factor=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, target_size)
            img_brightened = cv2.convertScaleAbs(img_resized, alpha=brightness_factor, beta=0)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_brightened)
            print(f"Processed: {filename}")

# Image enhancement: Histogram Equalization and Denoising
def histogram_equalization(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        enhanced = cv2.equalizeHist(image)
    return enhanced

def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Apply histogram equalization to images
def apply_histogram_equalization(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for image_path in Path(input_dir).glob("*.[jp][pn]g"):
        image = cv2.imread(str(image_path))
        enhanced_image = histogram_equalization(image)
        enhanced_image = denoise(enhanced_image)
        output_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(output_path), enhanced_image)
    print(f"Histogram equalization applied and results saved to: {output_dir}")

# Multi-Scale Retinex (MSR)
def singleScaleRetinex(img, variance):
    retinex = np.log10(img + 1) - np.log10(cv2.GaussianBlur(img, (0, 0), variance) + 1)
    return retinex

def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        zero_count = count[np.where(unique == 0)[0][0]]
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex

# Apply Retinex Single Scale Retinex (SSR)
def apply_retinex_ssr(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_file in os.listdir(input_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path)
            image_float = image.astype(np.float32)
            retinex_ssr = np.log1p(image_float) - np.log1p(cv2.GaussianBlur(image_float, (0, 0), 25))
            retinex_ssr = (retinex_ssr - np.min(retinex_ssr)) / (np.max(retinex_ssr) - np.min(retinex_ssr)) * 255
            retinex_ssr = retinex_ssr.astype(np.uint8)
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, retinex_ssr)
            print(f"Processed: {image_file}")

# Nonlinear gray transformation
def nonlinear_gray_transformation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = np.power(gray_image / 255.0, 0.5) * 255.0
    transformed_image = transformed_image.astype(np.uint8)
    return transformed_image

# Linear gray transformation
def linear_gray_transformation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = cv2.convertScaleAbs(gray, alpha=1.0, beta=25)
    return transformed_image

# Define directories and parameters for each operation
directories = [
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_cmyk", "operation": "convert_to_cmyk"},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_resize1", "operation": "preprocess_images", "params": {"target_size": (32, 32), "brightness_factor": 1.5}},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_enhancedhistogram", "operation": "apply_histogram_equalization"},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_histogram_equalized", "operation": "apply_histogram_equalization"},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_MSR", "operation": "MSR", "params": {"variance_list": [15, 80, 30]}},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_Retinex_SSR", "operation": "apply_retinex_ssr"},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_nonlinear", "operation": "nonlinear_gray_transformation"},
    {"input": "/home/student/alaaabo/data/images/train", "output": "/home/student/alaaabo/data/images/train_linear1", "operation": "linear_gray_transformation"}
]

# Execute the respective operations
for directory in directories:
    input_dir = directory["input"]
    output_dir = directory["output"]
    operation = directory["operation"]
    params = directory.get("params", {})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if operation == "convert_to_cmyk":
                cmyk_img = convert_to_cmyk(image_path)
                if cmyk_img:
                    cmyk_img.save(output_path)
                    print(f"Converted and saved {output_path}")

            elif operation == "preprocess_images":
                img = cv2.imread(image_path)
                img_resized = cv2.resize(img, params["target_size"])
                img_brightened = cv2.convertScaleAbs(img_resized, alpha=params["brightness_factor"], beta=0)
                cv2.imwrite(output_path, img_brightened)
                print(f"Processed: {filename}")

            elif operation == "apply_histogram_equalization":
                image = cv2.imread(image_path)
                enhanced_image = histogram_equalization(image)
                enhanced_image = denoise(enhanced_image)
                cv2.imwrite(output_path, enhanced_image)

            elif operation == "MSR":
                img = cv2.imread(image_path)
                img_msr = MSR(img, params["variance_list"])
                cv2.imwrite(output_path, img_msr)
                print(f"Processed and saved {filename}")

            elif operation == "apply_retinex_ssr":
                img = cv2.imread(image_path)
                img_float = img.astype(np.float32)
                retinex_ssr = np.log1p(img_float) - np.log1p(cv2.GaussianBlur(img_float, (0, 0), 25))
                retinex_ssr = (retinex_ssr - np.min(retinex_ssr)) / (np.max(retinex_ssr) - np.min(retinex_ssr)) * 255
                retinex_ssr = retinex_ssr.astype(np.uint8)
                cv2.imwrite(output_path, retinex_ssr)
                print(f"Processed: {filename}")

            elif operation == "nonlinear_gray_transformation":
                img = cv2.imread(image_path)
                transformed_image = nonlinear_gray_transformation(img)
                cv2.imwrite(output_path, transformed_image)
                print(f"Transformed {filename}")

            elif operation == "linear_gray_transformation":
                img = cv2.imread(image_path)
                transformed_image = linear_gray_transformation(img)
                cv2.imwrite(output_path, transformed_image)
                print(f"{filename} processed and saved.")

print("All processing complete.")
