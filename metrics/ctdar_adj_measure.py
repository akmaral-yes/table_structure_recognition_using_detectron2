import copy
import argparse
import os
import pandas as pd
from utils.draw_utils import draw_predictions_gt
from generate_adj_list import CtdarAdjacencyRelationsGenerator
from utils.file_utils import load_dict, save_dict
from utils.box_utils import compute_iou
from detectron2_tables.postproc_maskrcnn import postproc
from compare_adj_list import compare_adj_rel_lists
from build_cells import build_cells_from_columns_rows, combine_content_row_col
from build_cells import match_cell_row_col



def weighted_avg_f1(
        gt_table_structures, pred_table_structures, eval_path, debug
):
    """
    Compute weighted average F1-score
    (iou_thresh1*F1-score@iou_thresh1+..+iou_thresh4*F1-score@iou_thresh4)/
        sum(iou_thresh1+iou_thresh2+iou_thresh3+iou_thresh4)
    """
    columns = ["iou_thresh", "F1-score"]
    measures = pd.DataFrame()
    iou_threshs = [0.6, 0.7, 0.8, 0.9]
    f1_scores = []
    weighted_f1_scores = []
    for iou_thresh in iou_threshs:
        iou_thresh_str = "".join(str(iou_thresh).split("."))
        pred_table_structures_matched = match_pred_with_gt(
            gt_table_structures, pred_table_structures, iou_thresh
        )
        metrics_dict, f1_score = compare_adj_rel_lists(
            gt_table_structures, pred_table_structures_matched
        )
        if debug:
            save_dict(
                eval_path, "metrics_" + iou_thresh_str + ".json", metrics_dict
            )
        print("iou_thresh: ", iou_thresh, "  f1_score: ", f1_score)
        f1_scores.append(f1_score)
        weighted_f1_scores.append(iou_thresh * f1_score)
        measures = measures.append(pd.DataFrame(
                    [[iou_thresh, f1_score]], columns=columns))
    w_avg_f1 = sum(weighted_f1_scores) / sum(iou_threshs)
    measures = measures.append(pd.DataFrame(
                    [["w_avg_f1", w_avg_f1]], columns=columns))    
    return w_avg_f1, measures


def match_pred_with_gt(gt_table_structures, pred_table_structures, iou_thresh):
    """
    Replace cells in predicted table structures with the ones from
    ground-truth table structures if their intersection over union at least
    equal to iou_thresh
    """
    pred_table_structures_copy = copy.deepcopy(pred_table_structures)
    for table_name, pred_table_structure in pred_table_structures_copy.items():
        gt_table_structure = gt_table_structures[table_name]
        # Build a list of all ground-truth cells for the given table
        gt_cells = []
        for cell1, cell2, _, _ in gt_table_structure:
            if cell1 not in gt_cells:
                gt_cells.append(cell1)
            if cell2 not in gt_cells:
                gt_cells.append(cell2)
        # Compare IoU of the predicted cells with every ground-truth cells
        for idx, pred_adj_relationship in enumerate(pred_table_structure):
            cell1 = pred_adj_relationship[0]
            cell2 = pred_adj_relationship[1]
            for cell in gt_cells:
                if compute_iou(cell, cell1) >= iou_thresh:
                    pred_table_structure[idx][0] = cell
                if compute_iou(cell, cell2) >= iou_thresh:
                    pred_table_structure[idx][1] = cell
        pred_table_structures_copy[table_name] = pred_table_structure
    return pred_table_structures_copy


def predict_table_structures(input_path, images_path, spanning):
    """
    Post-process row and column predictions, build cell predictions from
    them
    """
    row_ctdar_preds_dict = load_dict(input_path, 'row_ctdar_preds_dict.json')
    row_ctdar_preds_dict = postproc(
        images_path, row_ctdar_preds_dict, dataclass="row", datatype="ctdar",
        annotations="", post_path="", postprocess="rule", thresh=0.7
    )

    column_ctdar_preds_dict = load_dict(
        input_path, 'column_ctdar_preds_dict.json'
    )
    column_ctdar_preds_dict = postproc(
        images_path, column_ctdar_preds_dict, dataclass="column",
        datatype="ctdar", annotations="", post_path="", postprocess="rule",
        thresh=0.7
    )

    cells_pred_dict = build_cells_from_columns_rows(
        row_ctdar_preds_dict, column_ctdar_preds_dict,
        table_images = images_path
    )
    full_cell_pred_dict = load_dict(
        input_path, "content_ctdar_preds_dict.json"
    )
    full_cell_pred_dict = postproc(
        images_path, full_cell_pred_dict, dataclass="cell", datatype="ctdar",
        annotations="", post_path="", postprocess="rule", thresh=0.6
    )

    if spanning:
        # Combine direct cell predictions with cell predicitons from columns
        # and rows

        cells_pred_dict = match_cell_row_col(
            cells_pred_dict, full_cell_pred_dict
        )

    cells_pred_dict = combine_content_row_col(
        cells_pred_dict, full_cell_pred_dict
    )

    return cells_pred_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="../datasets/ctdar",
                        help="A path to a folder with gt_tables_dict.json,\
                        column_ctdar_preds_dict.json,\
                        row_ctdar_preds_dict.json,\
                        content_ctdar_preds_dict.json,\
                        table_images")
    parser.add_argument('--output_path', default="../datasets/ctdar",
                    help="A path to save all intermediate files")
    parser.add_argument('--spanning', action='store_true',
                        help="Build spanning cells by combining row and column\
                        predictions with content predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Save all intermediate computations")
    args = parser.parse_args()

    output_path = args.output_path
    input_path = args.input_path
    table_images_path = os.path.join(input_path, "table_images")
    eval_path = os.path.join(output_path, "evaluation")
    os.makedirs(eval_path, exist_ok=True)

    gt_tables_dict = load_dict(input_path, "ctdar_gt_dict.json")
    adj_list_generator = CtdarAdjacencyRelationsGenerator()
    gt_table_strs = adj_list_generator.generate_adj_rel(gt_tables_dict)
    save_dict(eval_path, "gt_table_strs.json", gt_table_strs)

    cells_pred_dict = predict_table_structures(
        input_path, table_images_path, args.spanning
    )
    if args.debug:
        save_dict(eval_path, "cells_pred_dict.json", cells_pred_dict)

    pred_table_strs = adj_list_generator.generate_adj_rel(cells_pred_dict)
    if args.debug:
        save_dict(eval_path, "pred_table_strs.json", pred_table_strs)

        tables_predictions_gt_path = os.path.join(eval_path, "tables_preds_gt")
        os.makedirs(tables_predictions_gt_path, exist_ok=True)
        draw_predictions_gt(table_images_path, gt_tables_dict, cells_pred_dict,
                            tables_predictions_gt_path)

    w_avg_f1, measures = weighted_avg_f1(
        gt_table_strs, pred_table_strs, eval_path, args.debug
    )
    measures_path = os.path.join(eval_path, "weighted_avg_f1.csv")
    measures.to_csv(measures_path, index=False)
    print("weighted average F1-score: ", w_avg_f1)


if __name__ == "__main__":
    main()
