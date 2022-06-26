__author__ = "Lingyue Fu"
__version__ = "1.0"

import cv2
import numpy as np


def opencv_inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


def ipbasic_inpaint(
    depth_map, extrapolate=False, blur_type="bilateral", show_process=False
):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    """
    Some Params
        max_in: max depth value (real)
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for depths > 0.375 * max_in
        dilation_kernel_med: dilation kernel to use for 0.1875 * max_in < depths < 0.375 * max_in
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 0.1875 * max_in
    """
    max_depth = depth_map.max() + 200

    # Full kernels
    FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
    FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
    FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
    FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
    FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

    # 3x3 cross kernel
    CROSS_KERNEL_3 = np.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # 5x5 cross kernel
    CROSS_KERNEL_5 = np.asarray(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    # 5x5 diamond kernel
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    # 7x7 cross kernel
    CROSS_KERNEL_7 = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # 7x7 diamond kernel
    DIAMOND_KERNEL_7 = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    dilation_kernel_far = CROSS_KERNEL_3
    dilation_kernel_med = CROSS_KERNEL_5
    dilation_kernel_near = CROSS_KERNEL_7

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    max_in = depths_in.max()
    # valid_pixels_near = (depths_in > 0.1) & (depths_in <= 0.1875 * max_in)
    # valid_pixels_med = (depths_in > 0.1875 * max_in) & (depths_in <= 0.375 * max_in)
    # valid_pixels_far = (depths_in > 0.375 * max_in)

    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = depths_in > 30.0

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = s1_inverted_depths > 0.1
    s1_inverted_depths[valid_pixels] = max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far), dilation_kernel_far
    )
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med), dilation_kernel_med
    )
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near), dilation_kernel_near
    )

    # Find valid pixels for each binned dilation
    valid_pixels_near = dilated_near > 0.1
    valid_pixels_med = dilated_med > 0.1
    valid_pixels_far = dilated_far > 0.1

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5
    )

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = s3_closed_depths > 0.1
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = s4_blurred_depths > 0.1
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[
        top_row_pixels, range(s5_dilated_depths.shape[1])
    ]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[
                0 : top_row_pixels[pixel_col_idx], pixel_col_idx
            ] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == "gaussian":
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == "bilateral":
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict["s0_depths_in"] = depths_in

        process_dict["s1_inverted_depths"] = s1_inverted_depths
        process_dict["s2_dilated_depths"] = s2_dilated_depths
        process_dict["s3_closed_depths"] = s3_closed_depths
        process_dict["s4_blurred_depths"] = s4_blurred_depths
        process_dict["s5_combined_depths"] = s5_dilated_depths
        process_dict["s6_extended_depths"] = s6_extended_depths
        process_dict["s7_blurred_depths"] = s7_blurred_depths
        process_dict["s8_inverted_depths"] = s8_inverted_depths

        process_dict["s9_depths_out"] = depths_out

    return depths_out, process_dict
