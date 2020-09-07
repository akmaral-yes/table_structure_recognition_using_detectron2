import cv2
import os
import random
from utils.file_utils import load_dict
from utils.preproc_utils import binarization
from utils.box_utils import bounding_rects_comp


def draw_cell_borders(table_name, cells_list, tables_path, gt_cells_path):
    """
    Draw cell borders with green color and save the images
    """
    table_path = os.path.join(tables_path, table_name)
    if len(cells_list) != 0:
        img = cv2.imread(table_path)
        img = draw_rectangles_xywh(img, cells_list, color=(0, 255, 0))
        cv2.imwrite(os.path.join(gt_cells_path, table_name), img)


def draw_rectangles_xywh(img, rect_list, color, thickness=3):
    """
    Given a list of rectangles draw them with a given color
    """
    for x, y, w, h in rect_list:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    return img


def draw_lines(table_name, tables_path, gt_rows_cols_path, gt_dict_lines):
    """
    Draw horizontal and vertical lines
    """
    table_path = os.path.join(tables_path, table_name)
    img = cv2.imread(table_path)
    vertical_lines = gt_dict_lines[table_name].vertical_lines
    horizontal_lines = gt_dict_lines[table_name].horizontal_lines
    for l in vertical_lines:
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 4, cv2.LINE_AA)
    for l in horizontal_lines:
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imwrite(os.path.join(gt_rows_cols_path, table_name), img)


def draw_rectangles(img, rect_list, color, thickness=1):
    """
    Given a list of rectangles draw them with given color
    """
    for x1, y1, x2, y2 in rect_list:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_polys(img, polys, color):
    """
    Args:
        img: cv2 image
        polys: a list of polys that needed to be drawn
                ([x1, y1, x2, y1, x2, y2, x1, y2])
        color: RGB code
    Returns:
        the image with drawn rectangles filled with a given color
    """
    for poly in polys:
        x1 = poly[0]
        y1 = poly[1]
        x2 = poly[2]
        y2 = poly[-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img


def draw_filled_rect(tables_path, tables_rect_path):
    """
    Binarize table images and draw filled rectangles over small connected
    components
    """
    images_list = os.listdir(tables_path)
    for table_name in images_list:
        table_path = os.path.join(tables_path, table_name)
        table_image = cv2.imread(table_path, 0)
        table_binary = binarization(table_image)
        table_binary = cv2.bitwise_not(table_binary)
        rect_list = bounding_rects_comp(table_binary)
        color = (0, 0, 0)
        draw_rectangles(table_binary, rect_list, color, -1)
        table_rect_path = os.path.join(tables_rect_path, table_name)
        cv2.imwrite(table_rect_path, table_binary)


def draw_annotations(annotations_path, tables_path, tables_gt_path, mask=True):
    """
    Draw on the original table images the bounding-boxes of each column/row,
    and segmentation masks
    """
    names = os.listdir(annotations_path)
    for doc_name in names:
        annotations = load_dict(annotations_path, doc_name)
        for table_name, annotation in annotations.items():
            table_path = os.path.join(tables_path, table_name)
            print("table_path: ", table_path)
            table_image = cv2.imread(table_path)
            # Instance = column/row/cell
            number_of_instances = len(annotation)
            # Draw annotation column by column
            for i in range(0, number_of_instances):
                bbox = annotation[i]["bbox"]
                polys = annotation[i]["segmentation"]
                x1, y1, x2, y2 = bbox
                r = random.choice(list(range(200)))
                g = random.choice(list(range(200)))
                b = random.choice(list(range(200)))
                # A little bit differentiate bbox and mask colors
                color_bbox = (r, g, b)
                color_mask = (r + 30, g + 30, b + 30)

                cv2.rectangle(table_image, (x1, y1), (x2, y2), color_bbox, 3)
                if mask:
                    draw_polys(table_image, polys, color_mask)
            # Save table images with drawn ground truth
            table_gt_path = os.path.join(tables_gt_path, table_name)
            cv2.imwrite(table_gt_path, table_image)


def draw_predictions_gt(table_images_path, cells_gt_dict, cells_predicted_dict,
                        tables_predictions_gt_path):
    """
    Draw predicted and ground truth cells
    """
    table_names = os.listdir(table_images_path)
    for table_name in table_names:
        table_image_path = os.path.join(table_images_path, table_name)
        table_image = cv2.imread(table_image_path)
        cells_gt_list = cells_gt_dict[table_name]["cells_list"]
        color_gt = (0, 255, 0)
        #table_image = draw_rectangles(table_image, cells_gt_list, color_gt)
        cells_predicted_list = cells_predicted_dict[table_name]["cells_list"]
        color_predicted = (255, 0, 0)
        table_image = draw_rectangles(
            table_image, cells_predicted_list, color_predicted, 3)
        table_predictions_gt_path = os.path.join(
            tables_predictions_gt_path, table_name)
        cv2.imwrite(table_predictions_gt_path, table_image)


def draw_preds_annotations(table_images_path, annotations, preds_dict,
                           table_preds_gt_path, datatype):
    """
    Draw predicted and ground truth cells
    """
    table_names = os.listdir(table_images_path)
    for table_name in table_names:
        table_image_path = os.path.join(table_images_path, table_name)
        table_image = cv2.imread(table_image_path)
        if datatype in  ['real', 'real3']:
            instances = annotations[table_name]
            bbox_gt = [instance["bbox"] for instance in instances]
        elif datatype in ['icdar', 'ctdar', 'unlv']:
            bbox_gt = annotations[table_name]['cells_list']
        #color_gt = (0, 255, 0)
        #table_image = draw_rectangles(table_image, bbox_gt, color_gt)
        preds_list = preds_dict[table_name]["bbox_predictions"]
        color_predicted = (255, 0, 0)
        table_image = draw_rectangles(table_image, preds_list, color_predicted, 3)
        table_pred_gt_path = os.path.join(table_preds_gt_path, table_name)
        cv2.imwrite(table_pred_gt_path, table_image)
