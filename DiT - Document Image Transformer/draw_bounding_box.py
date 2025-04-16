import cv2
import numpy as np
import argparse

def draw_bounding_box(image_path, bbox_coordinates, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on an image using given coordinates.
    
    Parameters:
    image_path (str): Path to the input image
    bbox_coordinates (list): List of [x1, y1, x2, y2] coordinates
    color (tuple): BGR color tuple for the box (default: green)
    thickness (int): Thickness of the bounding box lines
    
    Returns:
    numpy.ndarray: Image with drawn bounding box
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox_coordinates)
    
    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    return image

def save_image_with_bbox(image_path, output_path, bbox_coordinates):
    """
    Save the image with drawn bounding box.
    
    Parameters:
    image_path (str): Path to the input image
    output_path (str): Path where the output image will be saved
    bbox_coordinates (list): List of [x1, y1, x2, y2] coordinates
    """
    # Draw bounding box on the image
    image_with_bbox = draw_bounding_box(image_path, bbox_coordinates)
    
    # Save the result
    cv2.imwrite(output_path, image_with_bbox)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding box on an image and save the result.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("output_path", type=str, help="Path where the output image will be saved")
    parser.add_argument("bbox_coordinates", type=float, nargs=4, help="Bounding box coordinates [x1, y1, x2, y2]")
    
    args = parser.parse_args()
    
    save_image_with_bbox(args.image_path, args.output_path, args.bbox_coordinates)