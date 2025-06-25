# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
import random
import argparse
from pathlib import Path
from functools import cmp_to_key
from collections import deque
import matplotlib.pyplot as plt

# --- Constants ---
MIN_CELL_AREA = 500
MAX_CELL_AREA = 53000
DISTANCE_THRESHOLD = 15
TOTAL_RELEVENT_BOXES = 4 + 13 + 21 * 12 + 1
HEADER = 4 + 13

# --- Helper Functions ---

def save_step_image(image, title, output_folder, filename, show_plot=False):
    """
    Helper function to save an image with a title bar at the top.
    This makes each step's output self-explanatory.
    """
    if len(image.shape) == 2:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_to_save = image.copy()

    border_size = 50
    image_with_border = cv2.copyMakeBorder(image_to_save, top=border_size, bottom=10, left=10, right=10,
                                           borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.putText(image_with_border, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    full_path = os.path.join(output_folder, filename)
    cv2.imwrite(full_path, image_with_border)

    if show_plot:
      plt.figure(figsize=(10,12))
      plt.imshow(cv2.cvtColor(image_with_border, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.title(title)
      plt.show()

def get_last_n_and_resort(contours, N=TOTAL_RELEVENT_BOXES, K=HEADER, y_threshold=DISTANCE_THRESHOLD):
    """
    Performs the original multi-stage sorting and selection of contours.
    """
    if not contours or N <= 0:
        return []
    contours_with_boxes = [(c, cv2.boundingRect(c)) for c in contours]
    sorted_all_contours = sorted(contours_with_boxes, key=lambda item: (item[1][1], item[1][0]))

    num_contours = len(sorted_all_contours)
    start_index = max(0, num_contours - N)
    last_n_group = sorted_all_contours[start_index:]

    if not last_n_group:
        return []

    actual_k = min(K, len(last_n_group))
    first_k_group = last_n_group[:actual_k]
    second_nk_group = last_n_group[actual_k:]

    sorted_k_group = sorted(first_k_group, key=lambda item: (item[1][1], item[1][0]))
    final_k_contours = [contour for contour, box in sorted_k_group]

    final_nk_contours = []
    if second_nk_group:
        def custom_sort_logic(item1, item2):
            y1, y2 = item1[1][1], item2[1][1]
            x1, x2 = item1[1][0], item2[1][0]
            if abs(y1 - y2) < y_threshold:
                return x1 - x2
            else:
                return y1 - y2
        
        sorted_nk_group = sorted(second_nk_group, key=cmp_to_key(custom_sort_logic))
        final_nk_contours = [contour for contour, box in sorted_nk_group]

    final_nk_contours = final_nk_contours[:len(final_nk_contours)-1]
    final_k_contours = final_k_contours[:4]
    final_k_contours = sorted(final_k_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    return final_k_contours + final_nk_contours


# --- MAIN PUBLIC FUNCTION (MODIFIED) ---

def extract_cell_coordinates(input_image_np: np.ndarray, output_folder=None, show_plots=False):
    """
    Takes a form image as a NumPy array, detects table cells, and returns both
    the coordinates and the resized image that was used for detection.
    
    Args:
        input_image_np (np.ndarray): The input image in OpenCV (NumPy BGR) format.
        output_folder (str, optional): Path to save intermediate debug images.
        show_plots (bool, optional): If True, display final result with Matplotlib.

    Returns:
        tuple: A tuple containing:
            - list: A list of cell coordinates (x, y, w, h).
            - np.ndarray: The resized image on which coordinates were calculated.
    """
    if input_image_np is None:
        print("Error: Received an empty image in the extractor.")
        return [], None

    # This resize is the single source of truth for the canvas dimensions.
    resized_image = input_image_np

    if output_folder:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        save_step_image(resized_image, "0. Resized Image", output_folder, "step_0_resized.png")

    # The rest of the logic operates on `resized_image`
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 5, 2)
    if output_folder:
        save_step_image(binary, "1c. Adaptive Threshold", output_folder, "step_1c_binary_inverted.png")

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    grid_mask = cv2.add(detected_horizontal, detected_vertical)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, closing_kernel)
    
    inverted_mask = cv2.bitwise_not(grid_mask)
    raw_contours, _ = cv2.findContours(inverted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    areaLimit = [c for c in raw_contours if MIN_CELL_AREA < cv2.contourArea(c) < MAX_CELL_AREA]

    cleaned_contours = get_last_n_and_resort(areaLimit)
    
    # Prepare the final list of coordinates
    cell_coordinates = [cv2.boundingRect(c) for c in cleaned_contours]

    if output_folder:
        image_with_cells = resized_image.copy()
        for i, (x, y, w, h) in enumerate(cell_coordinates):
            cv2.rectangle(image_with_cells, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image_with_cells, str(i), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        save_step_image(image_with_cells, "4. Final Detected Cells", output_folder, "step_4_final_detected_cells.png", show_plot=show_plots)
    
    print(f"\nSuccessfully extracted {len(cell_coordinates)} cell coordinates.")
    
    # Return both the coordinates AND the resized image they belong to
    return cell_coordinates, resized_image


# This block allows the script to be run directly for testing.
if __name__ == '__main__':
    """
    Example of how to run this script directly for testing purposes.
    """
    parser = argparse.ArgumentParser(
        description="Extract tabular data from preprocessed image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_path", type=str, 
                        # required=True, 
                        help="Path to the input PNG image.",
                        nargs='?', const=1, default='notebooks/data/raw/image1.png')
    parser.add_argument("--output_path", type=str, 
                        # required=True, 
                        help="Path to save the processed PNG image.",
                        nargs='?', const=1, default='notebooks/data/detection_steps_for_image1')
    args = parser.parse_args()
    # print(args.input_path)
    INPUT_IMAGE_PATH = Path(args.input_path)
    OUTPUT_FOLDER = Path(args.output_path)

    print("--- Running Table Extractor in Test Mode ---")

    # INPUT_IMAGE_PATH = 'data/raw/image1.png'
    # OUTPUT_FOLDER = 'data/detection_steps_for_image1'

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"\nERROR: Test image not found at '{INPUT_IMAGE_PATH}'")
    else:
        # Load with OpenCV before calling the function
        input_image = cv2.imread(INPUT_IMAGE_PATH)
        
        # Call the refactored function
        detected_coordinates, _ = extract_cell_coordinates(
            input_image_np=input_image,
            output_folder=OUTPUT_FOLDER,
            show_plots=True
        )

        if detected_coordinates:
            print("\n--- Returned Coordinates (first 5) ---")
            for i, coords in enumerate(detected_coordinates[:5]):
                print(f"Cell {i}: {coords}")