import cv2
import os
import pickle
from detectron2.structures import BoxMode
from utils.file_utils import load_dict


def generate_empty_table_dicts(images_path):
    """
    Build empty annotations for table images that does not have ground-truth
    and will be used for predictions
    """
    table_dicts = []
    image_names = os.listdir(images_path)
    for image_name in image_names:
        print('image_name: ', image_name)
        image_dict = {}
        image_path = os.path.join(images_path, image_name)
        image_dict["file_name"] = image_path
        image_dict["image_id"] = image_name[:-4]
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        image_dict["height"] = height
        image_dict["width"] = width
        annotations = []
        annotation = {}
        # Bounding box is given in absolute value
        annotation["bbox_mode"] = BoxMode.XYXY_ABS
        # The category label
        annotation["category_id"] = 0
        annotation["segmentation"] = []
        annotation["bbox"] = [0, 0, 0, 0]
        annotations.append(annotation)
        image_dict["annotations"] = annotations
        table_dicts.append(image_dict)

    return table_dicts


def generate_table_dicts(mode, datatype, dataclass, input_path):
    """
    Augment existing annotations in .json for tables
    """
    annotations_path = os.path.join(
        input_path, datatype, mode + "_" + dataclass + "_annotations"
    )
    names = os.listdir(annotations_path)
    images_path = os.path.join(input_path, datatype, mode + "_tables")
    tables_dicts = []
    for name in names:
        annotations_dict = load_dict(annotations_path, name)
        for table_name, annotations in annotations_dict.items():
            table_dict = {}
            table_path = os.path.join(images_path, table_name)
            print(table_path)
            table_dict["file_name"] = table_path
            table_dict["image_id"] = table_name[:-4]
            image = cv2.imread(table_path)
            height, width = image.shape[:2]
            table_dict["height"] = height
            table_dict["width"] = width
            annotations_extended = []
            for annotation in annotations:
                # Bounding box is given in absolute value
                annotation["bbox_mode"] = BoxMode.XYXY_ABS
                annotations_extended.append(annotation)
            table_dict["annotations"] = annotations_extended
            tables_dicts.append(table_dict)
    save_table_dicts(input_path, mode, datatype, dataclass, tables_dicts)
    return tables_dicts


def generate_file_name(input_path, mode, datatype, dataclass):
    """
    Build path for saving a file with annotations
    Ex.:
        tables_real4/test_row_annotations.pickle
    """
    return os.path.join(
        input_path, datatype, mode + "_" + dataclass + "_annotations.pickle"
    )


def save_table_dicts(input_path, mode, datatype, dataclass, table_dicts):
    """
    Save generated annotations
    """
    filename = generate_file_name(input_path, mode, datatype, dataclass)
    print("filename: ", filename)
    with open(filename, 'wb') as handle:
        pickle.dump(table_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_table_dicts(mode, datatype, dataclass, input_path):
    """
    Generate/Load annotations
    """
    filename = generate_file_name(input_path, mode, datatype, dataclass)
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            table_dicts = pickle.load(handle)
    else:
        table_dicts = generate_table_dicts(
            mode, datatype, dataclass, input_path
            )
    return table_dicts
