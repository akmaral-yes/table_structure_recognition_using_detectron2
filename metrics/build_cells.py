import os
import cv2
from operator import itemgetter
from utils.box_utils import io1, overlap, area_computation
from utils.box_utils import intersection_boxes, box_in_box_precise


def build_cells_from_columns_rows(
    row_dict, column_dict, table_images="", sizes_dict={}
    ):
    """
    Build a list of cells from a list of columns and rows per table image
    Build a list of corresponding columns and rows indices
    """
    cells_dict = {}

    for table_name in row_dict.keys():
        rows = row_dict[table_name]["bbox_predictions"]
        columns = column_dict[table_name]["bbox_predictions"]
        if table_images:
            table_image_path = os.path.join(table_images, table_name)
            table_image = cv2.imread(table_image_path, 0)
            height, width = table_image.shape[:2]
        if sizes_dict:
            x1, y1, x2, y2 = sizes_dict[table_name]["loc"]
            height, width = y2 - y1, x2 - x1
        cells_list = []
        col_row_list = []
        rows.sort(key=itemgetter(1))
        columns.sort(key=itemgetter(0))
        for idx_row, row in enumerate(rows):
            for idx_col, column in enumerate(columns):
                box = intersection_boxes(row, column)
                if box:
                    if idx_row == 0:
                        box = [box[0], 0, box[2], box[3]]
                    elif idx_row == len(rows) - 1:
                        box = [box[0], box[1], box[2], height]
                    if idx_col == 0:
                        box = [0, box[1], box[2], box[3]]
                    elif idx_col == len(columns) - 1:
                        box = [box[0], box[1], width, box[3]]
                    cells_list.append(box)
                    # Pad with -1 for generating adjacency relations list
                    col_row_list.append((idx_col, idx_row, -1, -1))
        cells_dict[table_name] = {}
        cells_dict[table_name]["cells_list"] = cells_list
        cells_dict[table_name]["cells_col_row"] = col_row_list
    return cells_dict


def combine_content_row_col(cells_pred_dict, full_cell_pred_dict):
    """
    Args:
        cells_pred_dict: a dictionary of cells derived from column and row
            predictions
            {table_name:{"cells_list":..,"cells_col_row": ..},..}
        full_cell_pred_dict: a dictionary of cells' content position
            predictions
    Returns:
        cells_pred_dict: the cell position predictions derived from row and
            column predictions are replaced by content position predictions
    """
    for table_name, predictions in cells_pred_dict.items():
        cells_list = predictions['cells_list']
        col_row_list = predictions['cells_col_row']
        content_pos_list = full_cell_pred_dict[table_name]["bbox_predictions"]
        idx_to_delete = []
        for idx, cell in enumerate(cells_list):
            inside = 0
            for content in content_pos_list:
                if io1(content, cell) > 0:
                    cell_upd = content
                    inside += 1
            if inside == 1:
                cells_list[idx] = cell_upd
            if inside == 0:
                idx_to_delete.append(idx)
        idx_to_delete.sort(reverse=True)
        for idx in idx_to_delete:
            del cells_list[idx]
            del col_row_list[idx]
        cells_pred_dict[table_name]['cells_list'] = cells_list
        cells_pred_dict[table_name]['cells_col_row'] = col_row_list

    return cells_pred_dict


def build_content_for_preds(cells_pred_dict, cells_gt_dict):
    """
    Match content for predicted cells from contents of ground-truth cells.
    At least thresh_overlap of the ground-truth cell should be inside a
    predicted cell then the content of the predicted cell uses
    the content of the ground-truth cell
    """
    thresh_overlap = 0.5
    for table_name, cells in cells_pred_dict.items():
        cells_gt_list = cells_gt_dict[table_name]["cells_list"]
        contents_gt_list = cells_gt_dict[table_name]["contents_list"]
        cells_pred_list = cells["cells_list"]
        contents_pred_list = []
        for cell_pred in cells_pred_list:
            content_pred = ""
            for id_gt, cell_gt in enumerate(cells_gt_list):
                if overlap(cell_gt, cell_pred) >= thresh_overlap:
                    content_pred = content_pred + contents_gt_list[id_gt]
            contents_pred_list.append(content_pred)
        cells_pred_dict[table_name]["contents_list"] = contents_pred_list
    return cells_pred_dict


def match_cell_row_col(cells_pred_dict, full_cell_pred_dict):
    """
    Use direct cell predictions to fine-tune cells detected from row-column
    approach.
    Merge spanning cells:
    1. Iterate over cell predictions derived from columns and rows
    2. For every cell match the direct cell prediction that has
        the biggest overlap with the given cell
    3. Merge all cells that have biggest overlap with the same direct cell
        prediction
    """
    for table_name, cells in full_cell_pred_dict.items():
        print("table_name: ", table_name)
        full_cells_list = cells["bbox_predictions"]
        cells_list = cells_pred_dict[table_name]["cells_list"]
        col_row_list = cells_pred_dict[table_name]["cells_col_row"]
        full_cell_idx = []  # a list with cell indices from full_cells_list
        for cell in cells_list:
            full_cell_overlaps = []
            for idx, full_cell in enumerate(full_cells_list):
                inter = intersection_boxes(cell, full_cell)
                if inter:
                    area = area_computation(inter)
                    full_cell_overlaps.append((idx, area))
            if full_cell_overlaps:
                # Choose only one cell that has the biggest overlap
                full_cell_overlaps.sort(key=itemgetter(1), reverse=True)
                full_cell_overlap = full_cell_overlaps[0][0]
            else:
                # all cells that do not have any overlaps
                full_cell_overlap = -1
            full_cell_idx.append(full_cell_overlap)

        # Sort all lists by increasing indices in full_cell_idx
        full_cell_idx, cells_list, col_row_list = (list(t) for t in zip(
            *sorted(zip(full_cell_idx, cells_list, col_row_list))))

        # Merge cells that overlaps to the same spanning cell
        merge_idx = []
        full_idx_prev = 100000  # the max number of cells
        # Example: full_cell_idx: -1,-1,-1, 0,0,0,1,2,3,4...
        for idx, full_idx in reversed(list(enumerate(full_cell_idx))):
            if full_idx == full_idx_prev:  # more than one cell
                merge_idx.append(idx + 1)
            else:
                if merge_idx:
                    merge_idx.append(idx + 1)
                    x_list = []
                    y_list = []
                    cols = []
                    rows = []
                    # Remove the cells that will be merged from
                    # col_row_list and cells_list
                    for m_idx in merge_idx:
                        x_list.append(cells_list[m_idx][0])
                        x_list.append(cells_list[m_idx][2])
                        y_list.append(cells_list[m_idx][1])
                        y_list.append(cells_list[m_idx][3])
                        col = col_row_list[m_idx][0]
                        if col not in cols:
                            cols.append(col)
                        row = col_row_list[m_idx][1]
                        if row not in rows:
                            rows.append(row)
                        del cells_list[m_idx]
                        del col_row_list[m_idx]
                    # Add the merged cell to col_row_list and cells_list
                    rows.sort()
                    cols.sort()
                    start_row = rows[0]
                    start_col = cols[0]
                    end_row = -1
                    end_col = -1
                    if len(rows) > 1:
                        end_row = rows[-1]
                    if len(cols) > 1:
                        end_col = cols[-1]
                    col_row_list.append(
                        (start_col, start_row, end_col, end_row)
                    )
                    x_min = min(x_list)
                    x_max = max(x_list)
                    y_min = min(y_list)
                    y_max = max(y_list)
                    cells_list.append([x_min, y_min, x_max, y_max])
                    merge_idx = []
            if full_idx == -1:  # cells that do not have any overlaps
                break
            full_idx_prev = full_idx
        cells_pred_dict[table_name]["cells_list"] = cells_list
        cells_pred_dict[table_name]["cells_col_row"] = col_row_list
    return cells_pred_dict
