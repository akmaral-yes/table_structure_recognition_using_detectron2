import os
import cv2
import argparse
from lxml import etree
from utils.file_utils import save_dict
from utils.draw_utils import draw_rectangles


def draw_cell_annotations(gt_tables_dict, tables_path, gt_tables_path):
    """
    Draw cells on table images
    """
    for table_name, table in gt_tables_dict.items():
        table_path = os.path.join(tables_path, table_name)
        table_image = cv2.imread(table_path)
        color = (0, 255, 0) # green
        draw_rectangles(table_image, table["cells_list"], color, thickness=1)
        gt_table_image_path = os.path.join(gt_tables_path, table_name)
        cv2.imwrite(gt_table_image_path, table_image)


def find_coords(obj, nsmap):
    """
    Given xml subtree retrieve Coords and convert them to int
    """
    coords = obj.find('Coords', nsmap)
    points = coords.get("points")
    points_list = points.split(" ")
    x1, y1 = points_list[0].split(",")
    x2, y2 = points_list[2].split(",")
    return [int(x1), int(y1), int(x2), int(y2)]


def retrieve_gt(xmls_path):
    """
    Retrieve table locations from xml files
    Args:
        xmls_path: a path to a folder with .xml files with ground truth
    Returns:
        gt_tables_dict: a dictionary with table locations,
            table_name = image_name + "_"+table_idx+".jpg"
            {table_name: [x1,y1,x2,y2]...}
    """
    xml_names = os.listdir(xmls_path)
    gt_tables_dict = {}
    for xml_name in xml_names:
        xml_path = os.path.join(xmls_path, xml_name)
        tree = etree.parse(xml_path)
        root = tree.getroot()
        for idx, table in enumerate(tree.findall('table', root.nsmap)):
            table_name = xml_name[:-4] + "_" + str(idx) + ".jpg"
            gt_tables_dict[table_name] = {}
            x1_t, y1_t, x2_t, y2_t = find_coords(table, root.nsmap)
            gt_tables_dict[table_name]["loc"] = x1_t, y1_t, x2_t, y2_t
            cells = table.findall('cell', root.nsmap)
            cells_list = []
            rows_cols_list = []
            for cell in cells:
                start_row = int(cell.get("start-row"))
                start_col = int(cell.get("start-col"))
                end_row = int(cell.get("end-row"))
                end_col = int(cell.get("end-col"))
                if end_col == start_col:
                    end_col = -1
                if end_row == start_row:
                    end_row = -1
                rows_cols_list.append([start_col, start_row, end_col, end_row])
                x1, y1, x2, y2 = find_coords(cell, root.nsmap)
                cells_list.append([x1 - x1_t, y1 - y1_t, x2 - x1_t, y2 - y1_t])
            gt_tables_dict[table_name]["cells_col_row"] = rows_cols_list
            gt_tables_dict[table_name]["cells_list"] = cells_list

    return gt_tables_dict


def crop_tables(gt_tables_dict, images_path, output_path):
    """
    Crops tables from page images
    Args:
        images_path: a path to a folder with images that contain tables
        gt_tables_dict: a dictionary with table positions,
            table_name = image_name + "_"+table_idx+".jpg"
            {table_name: [x1,y1,x2,y2]...}
        output_path: a path for a folder with saving cropped table images
    """
    table_images_path = os.path.join(output_path, "table_images")
    os.makedirs(table_images_path, exist_ok=True)
    for table_name, table in gt_tables_dict.items():
        x1, y1, x2, y2 = table["loc"]
        image_name = "_".join(table_name.split("_")[:2]) + ".jpg"
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        table_image = image[y1:y2, x1:x2]
        table_image_path = os.path.join(table_images_path, table_name)
        cv2.imwrite(table_image_path, table_image)
    return table_images_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../datasets/ctdar/_input',
                        help="A path with 'images' folder and 'xml' folder")
    parser.add_argument('--output_path', default='../datasets/ctdar',
                        help="A path to save all output: cropped tables and \
                        ground truth")
    parser.add_argument('--debug', action='store_true',
                        help="Draw ground truth")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    images_path = os.path.join(input_path, "images")
    xmls_path = os.path.join(input_path, "xml")

    gt_tables_dict = retrieve_gt(xmls_path)
    save_dict(output_path, "ctdar_gt_dict.json", gt_tables_dict)

    table_images_path = crop_tables(gt_tables_dict, images_path, output_path)

    if args.debug:
        # Draw ground truth
        gt_tables_path = os.path.join(output_path, "gt_table_images")
        os.makedirs(gt_tables_path, exist_ok=True)
        draw_cell_annotations(
            gt_tables_dict, table_images_path, gt_tables_path
            )


if __name__ == "__main__":
    main()
    # cTDaR_t10047.xml - ground truth does not have all cells
