import logging

import cv2
import numpy as np
import torch

from src.logger_utils import print_green, print_normal, print_red, print_yellow


def otsu_normal(img):
    """
    Applies Otsu's optimal thresholding algorithm to segment an image.

    Args:
        img (np.ndarray): The input grayscale image matrix.

    Returns:
        np.ndarray: The thresholded binary image containing only 0 and 255 values.
    """
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu


def adjust_brightness_contrast_cv(image, brightness=150, contrast=100):
    """
    Adjusts the overall brightness and contrast of an image utilizing linear scaling.

    Args:
        image (np.ndarray): The input target image.
        brightness (int, optional): The target lightness percentage. Defaults to 150.
        contrast (int, optional): The target contrast multiplication constraint. Defaults to 100.

    Returns:
        np.ndarray: The modified visual array constrained to standard bit depths.
    """
    # Convert percentages to OpenCV parameters
    alpha = contrast / 100.0  # Contrast control (1.0-3.0)
    beta = (brightness - 100) * 2.55  # Brightness control (0-100)

    # Apply contrast and brightness
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted


def preprocess_image(image_stage2):
    """
    Executes an optimized sequential preprocessing framework for microorganism delineation.

    Converts BGR scale down to grayscale, refines visualization with Otsu metrics,
    and applies combined cross-circular morphological kernels correctly resolving appendages.

    Args:
        image_stage2 (np.ndarray): The BGR initial image.

    Returns:
        np.ndarray: Assessed inverted binary layout with distinct background threshold gaps.
    """
    image_gray = cv2.cvtColor(image_stage2, cv2.COLOR_BGR2GRAY)

    # Apply adjustments (150% brightness, 200% contrast increase)
    adjusted_image = adjust_brightness_contrast_cv(
        image_gray, brightness=150, contrast=200
    )
    enhanced_rgb = otsu_normal(adjusted_image)

    # For small organisms (50-100μm):
    # Circular kernel (better preserves the shape)
    kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Cross kernel (good for thin appendages)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # Apply morphological operations to connect appendages while preserving shape
    # First dilation to connect appendages
    dilated = cv2.dilate(enhanced_rgb, kernel_cross, iterations=1)
    # Then erosion to restore original size
    eroded = cv2.erode(dilated, kernel_circular, iterations=1)
    # Convert to binary inverse: black objects (0) on white background (255)
    ret, preprocessed_image = cv2.threshold(eroded, 100, 255, cv2.THRESH_BINARY_INV)

    return preprocessed_image


def add_scale_bar(image, um_per_pixel=7.38, target_mm=1.0):
    """
    Annotates a dimensional scale bar dynamically formatted in visual proportionality.

    Args:
        image (np.ndarray): The native BGR/RGB presentation matrix.
        um_per_pixel (float, optional): Environmental scope calibrator. Defaults to 7.38.
        target_mm (float, optional): Desired rendering length annotation in mm. Defaults to 1.0.

    Returns:
        np.ndarray: Display-ready array embedding standardized scale rendering annotations.
    """
    # Create a copy to avoid modifying the original
    img_with_bar = image.copy()
    img_height, img_width = img_with_bar.shape[:2]

    # Calculate the width of the scale bar in pixels (FIXED based on physical resolution)
    # For 7.38 um/px: 1 mm = 1000 um / 7.38 um/px ≈ 136 pixels
    target_um = target_mm * 1000.0  # Convert mm to microns
    bar_width_px = int(target_um / um_per_pixel)

    # ========== DYNAMIC VISUAL SCALING ==========
    # Scale factor based on image width (reference: 1000px width)
    scale_factor = img_width / 1000.0

    # Dynamic bar properties (proportional to image size)
    bar_height = max(4, int(8 * scale_factor))  # Min 4px, scales with image
    padding = max(15, int(30 * scale_factor))  # Proportional padding from edges

    # Dynamic font properties
    base_font_scale = 1.5  # Base font scale for 1000px width
    font_scale = base_font_scale * scale_factor
    font_scale = max(0.4, font_scale)  # Minimum font scale for readability

    font_thickness = max(2, int(4 * scale_factor))  # Min thickness of 2

    # Colors
    bar_color = (255, 255, 255)  # White
    text_color = (255, 255, 255)  # White

    # Bar position (bottom-left corner with proportional padding)
    x1 = padding
    y1 = img_height - padding - bar_height
    x2 = x1 + bar_width_px
    y2 = img_height - padding

    # Draw the scale bar (filled rectangle)
    cv2.rectangle(img_with_bar, (x1, y1), (x2, y2), bar_color, -1)

    # Add text label
    text = f"{target_mm:.0f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size to center it above the bar
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )

    # Text position (centered above the bar with proportional spacing)
    text_x = x1 + (bar_width_px - text_width) // 2
    text_spacing = max(4, int(8 * scale_factor))  # Proportional spacing above bar
    text_y = y1 - text_spacing

    # Draw text with anti-aliasing
    cv2.putText(
        img_with_bar,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return img_with_bar


def convert_area_pixels_to_mm2(area_pixels, mm_per_pixel=0.00738):
    """
    Transforms computed regional pixel approximations onto realistic spatial units matching microscope limits.

    Args:
        area_pixels (float): Detected pixel accumulation per segmented contour.
        mm_per_pixel (float, optional): Multiplier ratio. Defaults to 0.00738.

    Returns:
        float: Estimated real biological area mapped into square millimeters (mm²).
    """
    area_mm2 = area_pixels * (mm_per_pixel**2)
    return area_mm2


def get_area_proxy_method(image_stage2, preprocessed_image):
    """
    Determines and extrapolates the predominant spatial mask and area using connected-components contouring.

    Processes binary inputs framing white borders ensuring organism integrity near bounding edges,
    suppressing parent backdrop contours systematically.

    Args:
        image_stage2 (np.ndarray): Spatial structure and geometric dimension references matrix.
        preprocessed_image (np.ndarray): Binarized object-highlighted input canvas.

    Returns:
        Tuple[bool, float, Optional[np.ndarray]]:
            - success (bool): Operation validity completion flag.
            - mask_area (float): Maximum physical area size corresponding to contour tracking.
            - mask_contour (np.ndarray): Fully filled single-channel uint8 proxy segment layer.
    """

    try:
        # Add a white border of 10 pixels on all sides to prevent body or antennaes from touching the edges
        bordered_image = cv2.copyMakeBorder(
            preprocessed_image,
            10,
            10,
            10,
            10,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(
            bordered_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            print_normal("No contours found after findContours()")
            return False, 0, None

        # Compute areas and remove the largest contour (assumed parent/background)
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            print_normal("Contour area list is empty")
            return False, 0, None

        largest_contour_idx = int(np.argmax(areas))
        # If there is only one contour, removing it leaves nothing -> handle that
        if len(contours) <= 1:
            print_normal(
                f"Only one contour present (index {largest_contour_idx}); nothing left after removal"
            )
            return False, areas[largest_contour_idx], None

        contours = [c for i, c in enumerate(contours) if i != largest_contour_idx]
        print_normal(
            f"Removed parent contour (index {largest_contour_idx}) with area: {areas[largest_contour_idx]:.2f} pixels"
        )

        # Select largest contour by area, supposed to be the organism
        largest_contour = max(contours, key=cv2.contourArea)

        # Adjust largest contour to account for the 10-pixel border (ensure it's a numpy array)
        largest_contour = np.asarray(largest_contour) - 10

        # Recompute areas for remaining contours and guard again
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            print_normal("No remaining contours after removing parent")
            return False, 0, None
        largest_contour_idx = int(np.argmax(areas))
        mask_area = float(areas[largest_contour_idx])

        # Create a masked image: black background with only the largest_contour area preserved
        # Ensure image_stage2 exists and has shape
        if "image_stage2" not in locals() and "image_stage2" not in globals():
            print_normal("image_stage2 is not defined in scope")
            return False, mask_area, None

        mask_contour = np.zeros(
            image_stage2.shape[:2], dtype=np.uint8
        )  # Only height and width, single channel
        cv2.drawContours(
            mask_contour, [largest_contour], -1, color=255, thickness=cv2.FILLED
        )

        return True, mask_area, mask_contour

    except cv2.error as e:
        print_normal(f"OpenCV error during contour processing: {e}")
        return False, 0, None
    except (ValueError, IndexError, TypeError, NameError) as e:
        print_normal(f"Unexpected error during contour processing: {e}")
        return False, 0, None


def crop_image_with_mask(image_tensor, mask_contour):
    """
    Applies a binary segmentation mask to the original image tensor to isolate the specimen.

    Args:
        image_tensor (Any): The incoming multi-channel image representation.
        mask_contour (np.ndarray): The 2D binary segmentation mask.

    Returns:
        Optional[torch.Tensor]: A masked image tensor matching original dimensions,
            or None if the binary mask equates to zero area.
    """
    # Check if image_tensor is already a numpy array with HWC format
    if isinstance(image_tensor, np.ndarray) and image_tensor.ndim == 3:
        img_to_crop_np = image_tensor
    else:
        img_to_crop_np = (
            image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        )  # CHW to HWC

    # Ensure mask_contour is binary (0 or 1)
    binary_mask_for_mult = (
        (mask_contour / 255).astype(np.uint8)
        if np.max(mask_contour) > 1
        else mask_contour.astype(np.uint8)
    )

    # Check if mask has any non-zero values
    if not np.any(binary_mask_for_mult):
        print_red("Mask is empty, skipping crop.")
        return None

    if img_to_crop_np.ndim == 3 and binary_mask_for_mult.ndim == 2:
        # Expand mask to have the same number of channels as the image for broadcasting
        masked_np = img_to_crop_np * binary_mask_for_mult[:, :, np.newaxis]
    elif img_to_crop_np.ndim == binary_mask_for_mult.ndim:  # Grayscale image case
        masked_np = img_to_crop_np * binary_mask_for_mult
    else:
        print_red("Image and mask dimensions are incompatible.")
        return None

    # Convert back to PyTorch tensor CHW (keeping original dimensions)
    masked_tensor = torch.from_numpy(masked_np).permute(2, 0, 1).float()
    return masked_tensor


def is_border_touch_acceptable(
    mask_contour: np.ndarray, tolerance_percent: float = 10.0
) -> bool:
    """
    Analyzes whether the segmented organism touches an image boundary, and if so,
    ensures it only touches one boundary and within an acceptable maximum limit.

    Args:
        mask_contour (np.ndarray): Binary mask (uint8) of the plankton organism.
        tolerance_percent (float): Margin of boundary intersection tolerated (e.g., 10 for 10%).

    Returns:
        bool: True if boundary contact is acceptable or non-existent, False if overly truncated.
    """
    h, w = mask_contour.shape
    max_pixels = int((tolerance_percent / 100.0) * w)  # para imagen cualquier size

    # Contar pixeles no-negros en los bordes
    top = np.count_nonzero(mask_contour[0, :])
    bottom = np.count_nonzero(mask_contour[-1, :])
    left = np.count_nonzero(mask_contour[:, 0])
    right = np.count_nonzero(mask_contour[:, -1])

    # Determinar cuántos bordes toca
    borders_touched = sum([top > 0, bottom > 0, left > 0, right > 0])

    # Aceptable: toca 0 o 1 borde y dentro del umbral
    if borders_touched == 0:
        return True
    if borders_touched == 1:
        return all(
            [
                top <= max_pixels,
                bottom <= max_pixels,
                left <= max_pixels,
                right <= max_pixels,
            ]
        )

    # Si toca más de un borde, lo descartamos
    return False
