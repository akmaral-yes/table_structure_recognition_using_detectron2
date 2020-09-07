import os
import argparse
import multiprocessing
import cv2
from connected_comps import find_connected_comp_in_cell, \
    find_connected_comp_in_colrow
from utils.file_utils import save_dict, load_dict
from utils.preproc_utils import binarization
from utils.draw_utils import draw_annotations, draw_filled_rect
from utils.box_utils import find_cells_in_box, build_boxes,\
    convert_list_xywh_to_xyxy


def find_bbox_polys(polys):
    """
    Find bounding-box over a list of polygons
    """
    if len(polys) == 0:
        return []
    bbox_x1 = min(poly[0] for poly in polys)
    bbox_y1 = min(poly[1] for poly in polys)
    bbox_x2 = max(poly[2] for poly in polys)
    bbox_y2 = max(poly[-1] for poly in polys)
    return [bbox_x1, bbox_y1, bbox_x2, bbox_y2]


def stretch_polys(polys, factor):
    """
    Stretch a list of polygons vertically by the given factor
    """
    polys_stretched = []
    for poly in polys:
        x1_t, y1_t, x2_t, y1_t, x2_t, y2_t, x1_t, y2_t = poly
        poly[1] = int(factor*poly[1])
        poly[3] = int(factor*poly[3])
        poly[5] = int(factor*poly[5])
        poly[7] = int(factor*poly[7])
        polys_stretched.append(poly)

    return polys_stretched


def stretch_box(box, factor):
    """
    Stretch the given box vertically by the given factor
    """
    return [box[0], int(factor*box[1]), box[2], int(factor*box[3])]


def generate_annotations(dataclass, factor, gt_tables_dict,
                         images_path, debug, images_prep_path):
    """
    Generate a part of annotations for the tables in one document that needed
    for detectron2 usage
    Args:
        dataclass: row/column
        gt_tables_dict: a path with cells positions (x1,y1,x2,y2)
        images_path: a path with table images(not stretched)
        images_prep_path: a path to save the binarized table images

    Returns:
        a part of input to detectron2
    """
    # Build a dictionary with all table images as a key
    tables_dict = {}
    for table_name, table in gt_tables_dict.items():
        print("table_name: ", table_name)
        cells = convert_list_xywh_to_xyxy(table["cells"])
        if dataclass in ["column", "row"]:
            # Depending on the dataclass use horizontal or vertical lines
            if dataclass == "column":
                lines = table["vertical_lines"]
            elif dataclass == "row":
                lines = table["horizontal_lines"]

            # Build a list of boxes from vertical/horizontal lines:
            # each box correspond to a column/row
            boxes_list = build_boxes(lines)
        else:
            boxes_list = cells
        image_path = os.path.join(images_path, table_name)
        image = cv2.imread(image_path, 0)

        # Preprocess image: binarize (white letters, black background)
        image_binary = binarization(image)

        if debug:
            # Save preprocessed image
            image_prep_path = os.path.join(images_prep_path, table_name)
            cv2.imwrite(image_prep_path, image_binary)

        annotations = []  # per table image

        for box in boxes_list:
            x1, y1, x2, y2 = box
            if dataclass in ["column", "row"]:
                # Crop the image of each column/row
                box_image = image_binary[y1:y2, x1:x2]
                # Find cells that are inside given column/row
                cells_inside = find_cells_in_box(cells, box)
                # Build segmentation mask as polygons and bbox over the mask
                polys = find_connected_comp_in_colrow(
                    box_image, x1, y1, cells_inside
                )
            elif dataclass in ["cell", "content"]:
                polys = find_connected_comp_in_cell(
                    image_binary, x1, y1, box
                    )
            # Compute the bounding-box position that contain all connected
            # components
            bbox = find_bbox_polys(polys)
            if len(bbox) == 0:
                continue
            annotation = {}
            if factor != 1:
                polys = stretch_polys(polys, factor)
                box = stretch_box(box, factor)
                bbox = stretch_box(bbox, factor)
            # Segmentation mask per column/row/cell
            annotation["segmentation"] = polys
            # Bounding box associated with the given column/row/cell
            if dataclass == "cell":
                annotation["bbox"] = box
            else:
                annotation["bbox"] = bbox
            # The class of columns/rows associated with 0
            annotation["category_id"] = 0
            annotations.append(annotation)
        tables_dict[table_name] = annotations
    return tables_dict


def generate_annotations_batch(multiproc,
                               dataclass,
                               factor,
                               gt_tables_dicts_path,
                               images_path,
                               annotations_path,
                               debug,
                               images_prep_path):
    """
    Generate annotations for a batch of documents
    """
    names = os.listdir(gt_tables_dicts_path)
    wrapper = WrapperGenerateAnnotations(
        dataclass,
        factor,
        gt_tables_dicts_path,
        images_path,
        annotations_path,
        debug,
        images_prep_path)

    if multiproc:
        print("multiproc")
        # Parallel run
        processes_number = multiprocessing.cpu_count()
        p = multiprocessing.Pool(processes_number)
        p.map(wrapper, names)
    else:
        # Sequential run
        for name in names:
            wrapper(name)


class WrapperGenerateAnnotations():

    def __init__(self, dataclass, factor, gt_tables_dicts_path, images_path,
                 annotations_path, debug, images_prep_path):
        self.dataclass = dataclass
        self.gt_tables_dicts_path = gt_tables_dicts_path
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.images_prep_path = images_prep_path
        self.debug = debug
        self.factor = factor

    def __call__(self, name):
        gt_tables_dict = load_dict(self.gt_tables_dicts_path, name)
        annotations = generate_annotations(
            self.dataclass,
            self.factor,
            gt_tables_dict,
            self.images_path,
            self.debug,
            self.images_prep_path)
        save_dict(self.annotations_path, name, annotations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../datasets/real',
                        help="A path to folders with table images \
                            (mode + '_tables') and ground-truth dicts \
                            (mode+_gt_tables_dict)")
    parser.add_argument('--output_path', default='../datasets/real',
                        help="A path to save all output")
    parser.add_argument('--dataclass', default='column',
                        help="Choose dataclass: row/column/cell/content")
    parser.add_argument('--mode', default='val',
                        help="Choose datatype train/val")
    parser.add_argument('--multiproc', action='store_true',
                        help="Use multiprocessing")
    parser.add_argument('--debug', action='store_true',
                        help="Debug: save prep images, draw annotations")
    parser.add_argument('--factor', default='1',
                        help="Vertical stretching factor")
    parser.add_argument('--draw_rect', action='store_true',
                        help="Draw filled rectangles over characters")

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    images_path = os.path.join(input_path, args.mode + "_tables")
    images_prep_path = ""

    annotations_path = os.path.join(output_path, args.mode + "_" +
                                    args.dataclass + "_annotations")
    os.makedirs(annotations_path, exist_ok=True)

    gt_tables_dicts_path = os.path.join(
        input_path, args.mode + "_gt_tables_dict")
    print(gt_tables_dicts_path)
    if args.debug:
        images_prep_path = os.path.join(output_path, args.mode + "_" + "prep")
        os.makedirs(images_prep_path, exist_ok=True)
    
    # Binarize images, generate detectron2 annotations
    generate_annotations_batch(
        args.multiproc,
        args.dataclass,
        float(args.factor),
        gt_tables_dicts_path,
        images_path,
        annotations_path,
        args.debug,
        images_prep_path)
    
    if args.debug:
        # Draw the generated annotations and save them
        # A path to save drawn annotations
        tables_gt_path = os.path.join(
            output_path, args.mode + "_" + args.dataclass + "_gt")
        os.makedirs(tables_gt_path, exist_ok=True)
        if float(args.factor) != 1:
            factor_str = args.factor
            factor_str = factor_str.replace(".", "_")
            images_path = os.path.join(
                input_path, args.mode + "_prep_tables_" + args.factor
                )
            print(images_path)
        draw_annotations(annotations_path, images_path, tables_gt_path)

    if args.draw_rect:
        # Draw filled rectangles
        tables_rect_path = os.path.join(output_path, args.mode + "_rect")
        os.makedirs(tables_rect_path, exist_ok=True)
        draw_filled_rect(images_path, tables_rect_path)


if __name__ == "__main__":
    main()
