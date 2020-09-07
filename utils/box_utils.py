import cv2


def build_boxes(lines):
    """
    Args:
        lines: horizontal or vertical lines
    Returns:
        a list of boxes: one box per row or column
    """
    boxes_list = []
    x1_prev, y1_prev, x2_prev, y2_prev = lines[0]
    for x1, y1, x2, y2 in lines[1:]:
        boxes_list.append((x1_prev, y1_prev, x2, y2))
        x1_prev, y1_prev = x1, y1

    return boxes_list


def box_in_box(box1, box2):
    """
    Check if box1 is inside box2
    """
    x1_left, y1_top, x1_right, y1_bottom = box1
    x2_left, y2_top, x2_right, y2_bottom = box2
    box = []
    if x1_left >= x2_right or x2_left >= x1_right:
        return box
    if y1_bottom <= y2_top or y2_bottom <= y1_top:
        return box
    x_left = max(x1_left, x2_left)
    x_right = min(x1_right, x2_right)
    y_top = max(y1_top, y2_top)
    y_bottom = min(y1_bottom, y2_bottom)
    box = [x_left, y_top, x_right, y_bottom]
    if (x_right - x_left <= 3) or (y_bottom - y_top <= 3):
        return []
    return box

def box_in_box_precise(box1, box2):
    """
    Check if box1 is inside box2
    """
    x1_left, y1_top, x1_right, y1_bottom = box1
    x2_left, y2_top, x2_right, y2_bottom = box2
    return x1_left>=x2_left and x1_right<=x2_right and y1_top>=y2_top and y1_bottom<=y2_bottom

def box_in_box_dataclass(box1, box2, dataclass):
    """
    If dataclass is "column": check nestedness in terms of horizontal
    positions, skip vertical positions
    If dataclass is "row": check nestedness in terms of vertical positions,
    skip horizontal positions
    If dataclass is "cell": check nestedness in terms of vertical positions
    and horizontal positions
    """
    x1_left, y1_top, x1_right, y1_bottom = box1
    x2_left, y2_top, x2_right, y2_bottom = box2
    if dataclass == "column":
        if x1_left >= x2_left and x1_right <= x2_right:
            return True
    elif dataclass == "row":
        if y1_bottom <= y2_bottom and y1_top >= y2_top:
            return True
    elif dataclass in ["cell", "content"]:
        if x1_left >= x2_left and x1_right <= x2_right and \
                y1_bottom <= y2_bottom and y1_top >= y2_top:
            return True
    return False


def find_cells_in_box(cells_list, bbox):
    """
    Returns:
        a list of cells inside the bbox, the cross-boundary cells are cutted
    """
    bbox_cells_list = []
    for cell in cells_list:
        box = box_in_box(cell, bbox)
        if len(box) != 0:
            bbox_cells_list.append(box)
    return bbox_cells_list


def intersection_boxes(box1, box2):
    """
    Returns:
        empty list: no intersection between boxes
        non-empty list: a box which is instersection of two boxes
    """
    x1_left, y1_top, x1_right, y1_bottom = box1
    x2_left, y2_top, x2_right, y2_bottom = box2
    box = []
    if x1_left >= x2_right or x2_left >= x1_right:
        return box
    if y1_bottom <= y2_top or y2_bottom <= y1_top:
        return box
    x_left = max(x1_left, x2_left)
    x_right = min(x1_right, x2_right)
    y_top = max(y1_top, y2_top)
    y_bottom = min(y1_bottom, y2_bottom)
    box = [x_left, y_top, x_right, y_bottom]
    return box


def compute_iou(box1, box2):
    """
    Compute intersection over union
    Args:
        box1:[x1,y1,x2,y2] top-left and bottom-right corners
        box2:[x1,y1,x2,y2] top-left and bottom-right corners
    """
    intersect_box = intersection_boxes(box1, box2)
    if len(intersect_box) == 0:
        return 0
    box1_area = area_computation(box1)
    box2_area = area_computation(box2)
    intersect_box_area = area_computation(intersect_box)

    return intersect_box_area / (box1_area + box2_area - intersect_box_area)


def io1(box1, box2):
    """
    Compute intersection over area of box1
    Args:
        box1:[x1,y1,x2,y2] top-left and bottom-right corners
        box2:[x1,y1,x2,y2] top-left and bottom-right corners
    """

    intersect_box = intersection_boxes(box1, box2)
    if len(intersect_box) == 0:
        return 0
    box1_area = area_computation(box1)
    if box1_area == 0:
        return 0
    intersect_box_area = area_computation(intersect_box)

    return intersect_box_area / (box1_area)


def bounding_rects_comp(img):
    """
    Find connected components of small sizes(exclude lines)
    Returns:
        a list of rectangles covering the connected components
    """
    # Invert an image: black background and white letters
    img_inv = cv2.bitwise_not(img)

    # Find connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_inv, None, None, None, 4, cv2.CV_32S
    )

    rect_list = []
    for i in range(0, nlabels - 1):
        width = stats[i, 2]
        height = stats[i, 3]
        area = stats[i, 4]
        # Filter out lines and big components
        if height / width < 0.2 or width / height < 0.2 or area > 1000:
            continue
        x_min = stats[i, 0]
        y_min = stats[i, 1]
        x_max = x_min + width
        y_max = y_min + height
        rect_list.append((x_min, y_min, x_max, y_max))
    return rect_list


def area_computation(box):
    """
    Computes the area inside box or 0 if empty box
    """
    if box:
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    else:
        return 0


def convert_list_xyxy_to_xywh(xyxy_list):
    """
    Convert a list of boxes with xyxy coordinates to the list of boxes with
    xywh coordinates
    """
    xywh_list = []
    for x1, y1, x2, y2 in xyxy_list:
        w = x2 - x1
        h = y2 - y1
        xywh_list.append((x1, y1, w, h))

    return xywh_list


def convert_list_xywh_to_xyxy(xywh_list):
    """
    Convert a list of boxes with xywh coordinates to the list of boxes with
    xyxy coordinates
    """
    return [[x1, y1, x1 + w, y1 + h] for [x1, y1, w, h] in xywh_list]


def extend_box(boxes, length, column=True):
    """
    Extend the predicted column/row to the whole height/width
    """
    boxes_ext = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if column:
            box_ext = [x1, 0, x2, length]
        else:
            box_ext = [0, y1, length, y2]
        boxes_ext.append(box_ext)
    return boxes_ext


def overlap_check(boxes):
    """
    Check if there is any overlapping boxes in the list
    """
    for box1 in boxes:
        for box2 in boxes:
            if box1 == box2:
                continue
            if intersection_boxes(box1, box2):
                return True
    return False


def overlap(box_gt, box_pred):
    """
    Args:
        box_gt: ground truth cell position (x1,y1,x2,y2)
        box_pred: predicted cell position (x1,y1,x2,y2)

    Returns:
        the percentage of the ground truth cell that is inside the
        predicted box intersection_area/ gt_cell_area
    """
    intersection_box = intersection_boxes(box_gt, box_pred)
    if intersection_box:
        intersection_area = (intersection_box[2] - intersection_box[0]) * (
            intersection_box[3] - intersection_box[1]
        )
        gt_area = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
        return intersection_area / gt_area
    return 0
