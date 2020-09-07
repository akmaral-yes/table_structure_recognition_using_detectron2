import cv2


def find_connected_comp_in_cell(img_inv, x, y, cell):
    """
    Args:
        img: a binary image with a white background and black letters
        x,y: top-left position of the cell relative to the table image
        cell (x1,y1,x2,y2): position of the cell in the original table image

    Returns:
        a list of the connected components inside given cell
    """
    # A list of the polygons found in the given image
    polys = []
    x1, y1, x2, y2 = cell
    # Crop the cell image
    cell_img_inv = img_inv[y1:y2, x1:x2]
    # Find connected components in the cell image
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cell_img_inv, None, None, None, 4, cv2.CV_32S
    )

    # Skip 0 as it corresponds to the background
    for i in range(1, nlabels):

        # (x_min, y_min) - top-left corner position
        # (x_max, y_max) - bottom-right corner position
        x_min, y_min, width, height, area = stats[i, :]
        x_max = x_min + width
        y_max = y_min + height
        
        # Skip gaussian noise
        if area <= 3:
            continue

        # Skip some dots in the top-left corner
        if area <= 7 and x_min == 0 and y_min == 0:
            continue
        
        # Skip if the component occupies the whole cell
        if x_min == 0 and y_min == 0 and width == x2 - x1 and \
                height == y2 - y1:
            continue
        
        # Skip a vertical line with height equal to the height of the cell,
        # and located on the left or right border of the cell
        if all([
            height >= 0.8 * (y2 - y1),
            float(width) / height <= 0.1,
            width <= 10,
            (x_min == 0 or x_max == x2 - x1),
        ]):
            continue

        # Skip a horizontal line with width equal to the width of the cell,
        # and located on the top or bottom border
        if all([
            width >= 0.8 * (x2 - x1),
            height / width <= 0.1,
            height <= 10,
            y_min == 0 or y_max == y2 - y1,
        ]):
            continue

        # Skip thin horizontal lines that attached to the bottom or to the
        # top of the cell
        if height / width <= 0.1 and height <= 2 and \
                (y_min == 0 or y_max == y2 - y1):
            continue
        
        # Position with respect to table image
        x1_t = x + int(x_min)
        y1_t = y + int(y_min)
        x2_t = x + int(x_max)
        y2_t = y + int(y_max)

        poly = [x1_t, y1_t, x2_t, y1_t, x2_t, y2_t, x1_t, y2_t]
        polys.append(poly)

    return polys


def find_connected_comp_in_colrow(img_inv, x, y, bbox_cells_list):
    """
    Args:
        img_inv: a binary image with a black background and white letters
        (x,y): top-left corner position of the column/row bbox in the original
            table image
        bbox_cells_list: a list of cells that are inside given column/row;
            (x1,y1,x2,y2) with respect to the table image

    Returns:
        a list of the connected components inside given column/row
    """

    # A list of the polygons found in the given image
    polys = []
    for x1, y1, x2, y2 in bbox_cells_list:
        # Cells' positions are given with respect to the table image
        # (top-left corner of the table image is (0,0))
        # Convert these positions with respect to the given column/row position
        # (top-left corner of the column/row is (0,0))
        cell = x1 - x, y1 - y, x2 - x, y2 - y

        polys.extend(find_connected_comp_in_cell(img_inv, x1, y1, cell))

    return polys
