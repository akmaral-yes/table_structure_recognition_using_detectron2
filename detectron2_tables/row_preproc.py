import os
import cv2
import argparse
import multiprocessing


def resize_images(tables_path, prep_tables_path, factor):
    """
    Stretch the images vertically by a given factor
    """
    table_names = os.listdir(tables_path)
    wrapper = WrapperResizeImage(tables_path, prep_tables_path, factor)
    processes_number = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes_number)
    p.map(wrapper, table_names)


def resize_image(table_name, tables_path, prep_tables_path, factor):
    """
    Stretch the image vertically by a given factor
    """
    table_path = os.path.join(tables_path, table_name)
    img = cv2.imread(table_path)
    width = img.shape[1]
    height = int(img.shape[0] * factor)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized_table_path = os.path.join(prep_tables_path, table_name)
    cv2.imwrite(resized_table_path, resized)


class WrapperResizeImage():

    def __init__(self, tables_path, prep_tables_path, factor):
        self.tables_path = tables_path
        self.prep_tables_path = prep_tables_path
        self.factor = factor

    def __call__(self, table_name):
        resize_image(
            table_name, self.tables_path, self.prep_tables_path, self.factor
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        default="../../6_icdar_dataset/output/table_images",
                        help="A path to a folder with table_images")
    parser.add_argument('--output_path',
                        default="../../6_icdar_dataset/output/",
                        help="A path for preprocessed images")
    parser.add_argument('--factor', default="1",
                        help="Choose scaling factor")
    args = parser.parse_args()

    factor_str = "".join(args.factor.split("."))
    prep_images_path = os.path.join(
        args.output_path, "prep_table_images_" + factor_str
    )
    os.makedirs(prep_images_path, exist_ok=True)
    resize_images(args.input_path, prep_images_path, float(args.factor))


if __name__ == "__main__":
    main()

