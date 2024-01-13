import math


def get_grid_pixel_groups(patch_size, image_size):
    """
    Generates a grid of rectangles of size patch_size to cover an image of size image_size with no padding.

    Args:
        patch_size: A tuple of two integers representing the size of the rectangle that will be used to patch the image
            and generate a rectangular grid
        image_size: A tuple of two integers representing the size of the image to be patched.

    Returns:
        A list of lists of integer pairs, each list of integer pairs representing pixel indices belonging to the same
            rectangular patch.
    """
    pixel_groups = []
    for i in range(math.ceil(image_size[0] / patch_size[0])):
        for j in range(math.ceil(image_size[1] / patch_size[1])):
            current_group = []
            for pixel_i_delta in range(patch_size[0]):
                for pixel_j_delta in range(patch_size[1]):
                    pixel_i = patch_size[0] * i + pixel_i_delta
                    pixel_j = patch_size[1] * j + pixel_j_delta
                    if pixel_i < image_size[0] and pixel_j < image_size[1]:
                        current_group.append((pixel_i, pixel_j))
            pixel_groups.append(current_group)

    return pixel_groups
