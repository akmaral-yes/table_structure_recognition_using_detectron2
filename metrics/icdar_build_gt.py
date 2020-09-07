import os
import argparse
import cv2
from PyPDF2 import PdfFileWriter, PdfFileReader
from wand.image import Image as WANDImage
from wand.color import Color
from lxml import etree
from utils.file_utils import save_dict, load_dict
from utils.draw_utils import draw_rectangles


def retrieve_images(dataset_path, output_path):
    """
    Convert .pdf to image and retrieve sizes of pdf pages
    Args:
        dataset_path: a path with the pdf documents
        output_path: a path where the 'images' folder will be created
    Returns:
        pdf_size_dict: a dictionary with page sizes
            {pdf_name: [width0, height0], [width1, height1]},..}
    """
    images_path = os.path.join(output_path, "images")
    os.makedirs(images_path, exist_ok=True)
    file_names = os.listdir(dataset_path)
    pdf_names = [name for name in file_names if name[-3:] == "pdf"]
    pdf_size_dict = {}
    for pdf_name in pdf_names:
        print("pdf_name: ", pdf_name)
        # Compute a list of every page size
        pdf_sizes = pdf_to_image(dataset_path, pdf_name, images_path)
        for idx, pdf_size in enumerate(pdf_sizes):
            width, height = pdf_size.upperRight
            pdf_sizes[idx] = round(width), round(height)
        pdf_size_dict[pdf_name] = pdf_sizes
        print(70 * "*")
    return pdf_size_dict


def pdf_to_image(input_dir, file_name, output_dir):
    """
    Convert given pdf to images
    Returns:
        pdf_sizes: a dictionary with page sizes
            {pdf_name: [width0, height0], [width1, height1]},..}
    """
    # Read the .pdf file
    file_path = os.path.join(input_dir, file_name)
    f = open(file_path, "rb")
    inputpdf = PdfFileReader(f)

    # Create temp_folder to save there temporatily separate pages of the .pdf
    temp_folder = "./temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    number_of_pages = inputpdf.numPages
    pdf_sizes = []
    RESOLUTION_COMPRESSION_FACTOR = 300
    # Iterate over all pages
    for i in range(number_of_pages):
        pdf_sizes.append(inputpdf.pages[i].mediaBox)
        # Split into single page pdf
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))

        with open(
            os.path.join(temp_folder, "document-page%i.pdf") % i, "wb"
        ) as outputStream:
            output.write(outputStream)

        # Open single page pdf as image, compress and save
        with WANDImage(
                filename=os.path.join(temp_folder, "document-page%i.pdf") % i,
                resolution=300) as img:
            img.background_color = Color("white")
            img.alpha_channel = 'remove'
            img.compression_quality = RESOLUTION_COMPRESSION_FACTOR
            img.save(filename=os.path.join(
                output_dir, file_name[:-4] + "_%i.png") % i
            )

        # Remove single page pdf
        os.remove(os.path.join(temp_folder, "document-page%i.pdf") % i)

    f.close()
    os.rmdir(temp_folder)
    return pdf_sizes


def crop_tables(dataset_path, output_path, pdf_size_dict):
    """
    Crop tables given coordinates in .xml
    """
    file_names = os.listdir(dataset_path)
    xml_names = [f[:-4] + "-reg.xml" for f in file_names if f[-3:] == "pdf"]
    table_images_path = os.path.join(output_path, "table_images")
    os.makedirs(table_images_path, exist_ok=True)
    sizes_dict = {}
    for xml_name in xml_names:
        print("xml_name: ", xml_name)
        xml_path = os.path.join(dataset_path, xml_name)
        tree = etree.parse(xml_path)
        root = tree.getroot()
        pdf_name = xml_name[:-8] + ".pdf"
        for table in tree.findall('table', root.nsmap):
            # Tables are enumerated starting from 1
            table_id = table.get("id")
            print("table_id: ", table_id)
            region = table.find("region", root.nsmap)

            # Pages are enumerated starting from 1
            page = region.get("page")
            print("page: ", page)
            bbox = region.find('bounding-box', root.nsmap)
            # bbox: (x1,y1) - bottom-left, (x2,y2) - top-right
            # from bottom-left corner of the page
            x1 = int(bbox.get("x1"))
            y1 = int(bbox.get("y1"))
            x2 = int(bbox.get("x2"))
            y2 = int(bbox.get("y2"))

            # Corrected ground-truth table position from 2 documents:
            # "us-006-reg.xml", "us-008-reg.xml"
            if xml_name == "us-008-reg.xml":
                y1 = y1 + 18
                y2 = y2 + 18
                x1 = x1 - 2
                x2 = x2 - 2
            elif xml_name == "us-006-reg.xml":
                y1 = y1 + 26
                y2 = y2 + 26
                x1 = x1 - 2
                x2 = x2 - 2

            # In pdf sizes, in a list enumeration starts from 0
            width_pdf, height_pdf = pdf_size_dict[pdf_name][int(page) - 1]

            image_name = xml_name[:-8] + "_" + str(int(page) - 1) + ".png"
            image_path = os.path.join(output_path, "images", image_name)
            print("image_path: ", image_path)
            table_name = xml_name[:-8] + "_" + table_id + ".png"
            image = cv2.imread(image_path)
            height_img, width_img = image.shape[:2]
            # height_img correspond to height_pdf,
            # width_img correspond to width_pdf
            if height_img < width_img:
                if height_pdf > width_pdf:
                    width_pdf, height_pdf = height_pdf, width_pdf

            # Compute (x1_img, y1_img) - top-left coord,
            # (x2_img, y2_img) - bottom-right coord counted from top-right
            # corner of the page image
            x1_img = round(x1 * width_img / width_pdf)
            x2_img = round(x2 * width_img / width_pdf)
            y1_img = round((height_pdf - y1) * height_img / height_pdf)
            y2_img = round((height_pdf - y2) * height_img / height_pdf)

            table_image = image[y2_img:y1_img, x1_img:x2_img]
            table_image_path = os.path.join(table_images_path, table_name)
            cv2.imwrite(table_image_path, table_image)
            sizes_dict[table_name] = {}
            sizes_dict[table_name]["img"] = (
                round(height_img), round(width_img)
            )
            sizes_dict[table_name]["pdf"] = (
                round(height_pdf), round(width_pdf)
            )
            sizes_dict[table_name]["loc"] = (
                x1_img, y2_img, x2_img, y1_img
            )
    return sizes_dict

def build_cells_gt(dataset_path, sizes_dict):
    files_list = os.listdir(dataset_path)
    str_xml_list = [
        f for f in files_list if f[-7:] == "str.xml" and f[-9] != "b"
    ]
    cells_gt_dict = {}
    for str_xml_name in str_xml_list:
        print("str_xml_name: ", str_xml_name)
        str_xml_path = os.path.join(dataset_path, str_xml_name)
        tree = etree.parse(str_xml_path)
        root = tree.getroot()
        for idx, table in enumerate(tree.findall('table', root.nsmap), 1):
            cells_list = []
            contents_list = []
            cells_col_row_list = []

            table_name = str_xml_name[:-8] + "_" + str(idx) + ".png"

            x, y, _, _ = sizes_dict[table_name]["loc"]
            height_image, width_image = sizes_dict[table_name]["img"]
            height_pdf, width_pdf = sizes_dict[table_name]["pdf"]

            if str_xml_name == "eu-015-str.xml":
                height_image, width_image = width_image, height_image
                height_pdf, width_pdf = width_pdf, height_pdf

            regions = table.findall('region', root.nsmap)
            for region in regions:
                col_increment = int(region.get("col-increment"))
                cells = region.findall('cell', root.nsmap)
                for cell in cells:
                    start_col = int(cell.get("start-col")) + col_increment
                    start_row = int(cell.get("start-row"))

                    end_col = cell.get("end-col")
                    if end_col is None:
                        end_col = -1
                    else:
                        end_col = int(end_col) + col_increment

                    end_row = cell.get("end-row")
                    if end_row is None:
                        end_row = -1
                    else:
                        end_row = int(end_row)

                    cells_col_row_list.append(
                        (start_col, start_row, end_col, end_row)
                    )

                    bbox = cell.find('bounding-box', root.nsmap)
                    # Coordinates from "-str.xml"
                    # bbox: (x1,y1) - bottom-left, (x2,y2) - top-right
                    x1 = int(bbox.get("x1"))
                    y1 = int(bbox.get("y1"))
                    x2 = int(bbox.get("x2"))
                    y2 = int(bbox.get("y2"))

                    # Correct ground-truth cell positions from 2 documents:
                    # "us-006-str.xml", "us-008-str.xml"
                    if str_xml_name == "us-008-str.xml":
                        y1 = y1 + 18
                        y2 = y2 + 18
                        x1 = x1 - 0.025 * x1
                        x2 = x2 - 0.025 * x2
                    elif str_xml_name == "us-006-str.xml":
                        y1 = y1 + 26
                        y2 = y2 + 26
                        x1 = x1 - 0.025 * x1
                        x2 = x2 - 0.025 * x2

                    # Coordinates with respect to a page image top-left corner
                    x1_img = round(x1 * width_image / width_pdf)  # left
                    x2_img = round(x2 * width_image / width_pdf)  # right
                    y1_img = round((height_pdf - y1) *
                                   height_image / height_pdf)  # bottom
                    y2_img = round((height_pdf - y2) *
                                   height_image / height_pdf)  # top

                    # Coordinates with respect to a table image top-left corner
                    x1_img = x1_img - x  # left
                    x2_img = x2_img - x  # right
                    y1_img = y1_img - y  # bottom
                    y2_img = y2_img - y  # top

                    # Cell content
                    content = cell.find('content', root.nsmap)

                    cells_list.append((x1_img, y2_img, x2_img, y1_img))
                    contents_list.append(content.text)

            cells_gt_dict[table_name] = {}
            cells_gt_dict[table_name]["cells_list"] = cells_list
            cells_gt_dict[table_name]["contents_list"] = contents_list
            cells_gt_dict[table_name]["cells_col_row"] = cells_col_row_list

    return cells_gt_dict


def draw_cells(table_images_path, cells_gt_dict, tables_gt_path):
    """
    Draw the borders of table cells on the table images
    """
    table_names = os.listdir(table_images_path)
    for table_name in table_names:
        table_image_path = os.path.join(table_images_path, table_name)
        table_image = cv2.imread(table_image_path)
        cells_list = cells_gt_dict[table_name]["cells_list"]
        table_image = draw_rectangles(table_image, cells_list, (0,255,0),3)
        table_gt_path = os.path.join(tables_gt_path, table_name)
        cv2.imwrite(table_gt_path, table_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        default='../datasets/icdar/_input/competition-dataset',
                        help="A path with files from competition-dataset-eu \
                            and competition-dataset-eu folders")
    parser.add_argument('--output_path', default='../datasets/icdar',
                        help="A path to save cropped tables")
    parser.add_argument('--debug', action='store_true',
                        help="Save the table images with cells")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Convert .pdfs to images and save the page sizes of pdfs
    pdf_size_dict = retrieve_images(args.input_path, args.output_path)
    save_dict(args.output_path, "pdf_size_dict.json", pdf_size_dict)

    # Crop table images from page images
    # pdf_size_dict = load_dict(args.output_path, "pdf_size_dict.json")
    sizes_dict = crop_tables(args.input_path, args.output_path, pdf_size_dict)
    save_dict(args.output_path, "sizes_dict.json", sizes_dict)

    table_images_path  = os.path.join(args.output_path, "table_images_")

    # Create a dictionary:
    # {table_name: cells_list, cells_col_row_col_row, contents_list;..}
    cells_gt_dict = build_cells_gt(args.input_path, sizes_dict)
    save_dict(args.output_path, "icdar_gt_dict.json", cells_gt_dict)

    # Draw the ground truth cell positions on the table images
    if args.debug:
        tables_gt_path = os.path.join(args.output_path, "gt_table_images")
        os.makedirs(tables_gt_path, exist_ok=True)
        draw_cells(table_images_path, cells_gt_dict, tables_gt_path)


if __name__ == "__main__":
    main()
