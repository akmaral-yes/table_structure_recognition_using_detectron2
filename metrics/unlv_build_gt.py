import os
import argparse
import cv2
from lxml import etree
from utils.file_utils import load_dict, save_dict
from utils.draw_utils import draw_rectangles


def check_file_names(images_path, gt_path):
    """
    Check if names in two folders are the same
    """
    image_names = os.listdir(images_path)
    image_names = [image_name[:-4] for image_name in image_names]
    image_names = set(image_names)
    print("len(image_names): ", len(image_names))

    gt_files = os.listdir(gt_path)
    gt_jsons = [gt_file[:-5] for gt_file in gt_files if gt_file[-4:] == "json"]
    gt_jsons = set(gt_jsons)
    print("len(gt_jsons): ", len(gt_jsons))

    intersection = image_names.intersection(gt_jsons)
    print("intersection: ", len(intersection))
    print("image_names- gt_jsons: ", image_names - gt_jsons)
    print("gt_jsons-image_names: ", gt_jsons - image_names)


def crop_tables(images_path, gt_path, tables_path):
    """
    Crops tables from page images
    Args:
        images_path: a path to a folder with images that contain tables
        gt_path: a path to a folder with .json gt of table locations in the
                images; .json: a list of dictionaries,
                example:
                [{"top": 946, "right": 2070, "left": 770, "bottom": 2973}]
        tables_path: a path for saving cropped table images
    """
    image_names = os.listdir(images_path)

    for image_name in image_names:
        image_path = os.path.join(images_path, image_name)
        img = cv2.imread(image_path)
        gt_name = image_name[:-4] + ".json"
        table_locs = load_dict(gt_path, gt_name)
        for idx, table_loc in enumerate(table_locs):
            y1 = table_loc["top"]
            y2 = table_loc["bottom"]
            x1 = table_loc["left"]
            x2 = table_loc["right"]
            table_image = img[y1:y2, x1:x2]
            table_path = os.path.join(
                tables_path, image_name[:-4] + "_" + str(idx) + ".png"
            )
            cv2.imwrite(table_path, table_image)


def retrieve_bboxes(table, nsmap, dataclass, loc):
    """
    Args:
        table: xml tree
        dataclass: column/row/cell
        loc: table coordinates in the page image
    Returns:
        a list of boxes(colums/row/cell): [[x1,y1,x2,y2],..] coordinates with
                        respect to the top-left corner of the table
    """
    boxes = []
    x_left, y_top, x_right, y_bottom = loc

    # Convert cell position with respect to the top-left corner
    cells_col_row = []
    for box in table.findall(dataclass, nsmap):
        x0 = int(box.get("x0")) - x_left
        x1 = int(box.get("x1")) - x_left
        y0 = int(box.get("y0")) - y_top
        y1 = int(box.get("y1")) - y_top
        boxes.append([x0, y0, x1, y1])
        if dataclass == "Cell":
            start_col = int(box.get("startCol"))
            start_row = int(box.get("startRow"))
            end_col = int(box.get("endCol"))
            end_row = int(box.get("endRow"))
            if end_col == start_col:
                end_col = -1
            if end_row == start_row:
                end_row = -1
            cells_col_row.append([start_col, start_row, end_col, end_row])
    if dataclass == "Cell":
        return boxes, cells_col_row

    # Column/Row in fact encoded in .xml as lines separating columns/rows
    # We want to get boxes over columns/rows
    lines = boxes
    boxes_from_lines = []
    line_prev = []
    x_prev, y_prev = -1, -1
    for idx, line in enumerate(lines):
        if idx != 0:
            if line_prev == line:
                # sometimes in ground truth of the same line encoded several
                # times
                continue
        x0, y0, x1, y1 = line
        if x0 == x1:  # vertical lines => columns
            if idx == 0:
                boxes_from_lines.append([0, 0, x0, y_bottom - y_top])
            else:
                boxes_from_lines.append([x_prev, 0, x0, y_bottom - y_top])
            if idx == len(lines) - 1:
                boxes_from_lines.append(
                    [x0, 0, x_right - x_left, y_bottom - y_top])
            x_prev = x0
        else:  # horizontal lines => rows
            if idx == 0:
                boxes_from_lines.append([0, 0, x_right - x_left, y0])
            else:
                boxes_from_lines.append([0, y_prev, x_right - x_left, y0])
            if idx == len(lines) - 1:
                boxes_from_lines.append(
                    [0, y0, x_right - x_left, y_bottom - y_top])
            y_prev = y0
        line_prev = line
    return boxes_from_lines, []


def retrieve_gt_col_row(gt_path):
    """
    Args: a path to a folder with xml ground-truth files
    Returns:
        a dictionary {table_name: {"loc":[x1,y1,x2,y2],
                                   "Column":[[x1,y1,x2,y2],.. ],
                                   "Row": [[x1,y1,x2,y2],.. ],
                                   "Cell": [[x1,y1,x2,y2],.. ]
                                    },...
                      }
    """
    file_names = os.listdir(gt_path)
    xml_names = [name for name in file_names if name[-3:] == "xml"]
    gt_dict = {}

    for xml_name in xml_names:
        xml_path = os.path.join(gt_path, xml_name)
        print("xml_path: ", xml_path)
        tree = etree.parse(xml_path)
        root = tree.getroot()
        tables = tree.find('Tables', root.nsmap)
        for idx, table in enumerate(tables.findall('Table', root.nsmap)):
            table_name = xml_name[:-4] + "_" + str(idx) + ".png"
            print("table_name: ", table_name)
            x_left = int(table.get("x0"))
            x_right = int(table.get("x1"))
            y_top = int(table.get("y0"))
            y_bottom = int(table.get("y1"))
            loc = [x_left, y_top, x_right, y_bottom]
            gt_dict[table_name] = {}
            gt_dict[table_name]["loc"] = loc
            for dataclass in ["Column", "Row", "Cell"]:
                boxes, cells_col_row = retrieve_bboxes(
                    table, root.nsmap, dataclass, loc)
                if dataclass == "Cell":
                    gt_dict[table_name]["cells_list"] = boxes
                    gt_dict[table_name]["cells_col_row"] = cells_col_row
                else:
                    if not len(boxes):
                        # only one column
                        boxes = [[0,0,x_right-x_left, y_bottom-y_top]]
                    gt_dict[table_name][dataclass.lower()] = boxes

    return gt_dict


def draw_annotations(gt_dict, tables_path, gt_images_path, dataclass):
    """
    Draw column/row/cell on table images
    """
    table_names = os.listdir(tables_path)
    for table_name in table_names:
        boxes = gt_dict[table_name][dataclass]
        table_path = os.path.join(tables_path, table_name)
        table_image = cv2.imread(table_path)
        table_image = draw_rectangles(
            table_image, boxes, (0, 255, 0), thickness=3)
        gt_image_path = os.path.join(gt_images_path, table_name)
        cv2.imwrite(gt_image_path, table_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../datasets/unlv/_input',
                        help="A path with 'images' folder and \
                        'unlv_xml_gt' folder")
    parser.add_argument('--output_path', default='../datasets/unlv',
                        help="A path to save all output: cropped tables and \
                        ground truth")
    parser.add_argument('--debug', action='store_true',
                        help="Draw ground truth")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    if 0:
        # Check which files are missing?
        images_path = os.path.join(input_path, "images")
        gt_path = os.path.join(input_path, "unlv_xml_gt")
        check_file_names(images_path, gt_path)

    # Crop tables => results in 558 tables
    images_path = os.path.join(input_path, "images")
    gt_path = os.path.join(input_path, "unlv_xml_gt")
    tables_path = os.path.join(output_path, "table_images")
    os.makedirs(tables_path, exist_ok=True)
    crop_tables(images_path, gt_path, tables_path)

    # Retrieve ground_truth from xml: column, row, loc
    gt_path = os.path.join(input_path, "unlv_xml_gt")
    gt_dict = retrieve_gt_col_row(gt_path)
    save_dict(output_path, "unlv_gt_dict.json", gt_dict)

    if args.debug:
        # Draw ground truth
        for dataclass in ["row", "column", "cells_list"]:
            gt_images_path = os.path.join(
                output_path, dataclass + "_gt_images"
            )
            os.makedirs(gt_images_path, exist_ok=True)
            draw_annotations(gt_dict, tables_path, gt_images_path, dataclass)


if __name__ == "__main__":
    main()
    # 6736_002.json does not exist, but 6736_002.png exists
    # 6737_002.png does not exist, but 6737_002.json exists
    # 6737_002.json renamed to 6736_002.json
    # 6737_002.xml renamed to 6736_002.xml
    # 558 tables in total
    # 9500_034.xml missing the cell positions
