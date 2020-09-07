import os
import argparse
from utils.file_utils import save_dict, load_dict
from utils.draw_utils import draw_predictions_gt
from metrics.icdar_build_row_col_indices import BuilderRowColIndices
from detectron2_tables.postproc_maskrcnn import postproc
from detectron2_tables.postproc_maskrcnn import postproc_stretched_pred
from metrics.build_cells import build_cells_from_columns_rows
from metrics.build_cells import match_cell_row_col, build_content_for_preds
from metrics.compare_adj_list import compare_adj_rel_lists, compare_docs
from metrics.generate_adj_list import IcdarAdjacencyRelationsGenerator


def rename_key(cell_dict):
    """
    Change key name from "bbox_predictions" to "cells_list"
    """
    for table_name in cell_dict.keys():
        cell_dict[table_name]["cells_list"] = cell_dict[table_name].pop(
            "bbox_predictions")
    return cell_dict


def predict_table_structures(
    input_path, eval_path, images_path, how, spanning, debug, factor
):
    """
    Post-process predictions and build cell predictions from them
    """
    datatype = "icdar"
    if how == "colrow":
        # Load predictions for columns and rows
        column_icdar_preds_dict = load_dict(
            input_path, "column_icdar_preds_dict.json"
        )
        row_icdar_preds_dict = load_dict(
            input_path, "row_icdar_preds_dict.json"
        )

        # Post-process predictions for columns
        column_icdar_preds_dict = postproc(
            images_path, column_icdar_preds_dict, "column",
            datatype, [], "", postprocess="rule", thresh=0.7
        )
        print("factor: ", factor)

        # Post-process predictions for rows
        row_icdar_preds_dict = postproc(
            images_path, row_icdar_preds_dict, "row", datatype, [],
            "", postprocess="rule", thresh=0.7, factor = factor
        )

        if debug:
            save_dict(eval_path, "column_icdar_pred_dict_ext.json",
                      column_icdar_preds_dict)
            save_dict(eval_path, "row_icdar_pred_dict_ext.json",
                      row_icdar_preds_dict)

        # Load a dictionary with image sizes
        sizes_dict = load_dict(input_path, "sizes_dict.json")

        # Build predictions for cells from column and row predictions
        cells_pred_dict = build_cells_from_columns_rows(
            row_icdar_preds_dict, column_icdar_preds_dict,
            sizes_dict = sizes_dict
        )
        if debug:
            save_dict(eval_path,
                        "cell_from_row_col_pred_dict.json", cells_pred_dict
                        )
        if spanning:
            # Combine direct cell predictions with cell predicitons derived
            # from columns and rows
            print("spanning: ", spanning)
            full_cell_pred_dict = load_dict(
                input_path, spanning+ "_icdar_preds_dict.json"
            )
            full_cell_pred_dict = postproc(
                images_path, full_cell_pred_dict, dataclass="content",
                datatype="icdar", annotations="", post_path="",
                postprocess="rule", thresh=0.7
            )
            cells_pred_dict = match_cell_row_col(
                cells_pred_dict, full_cell_pred_dict
            )

    elif how in ["cell", "content"]:
        # Load predictions for cells
        cell_icdar_pred_dict = load_dict(
            input_path, how + "_icdar_preds_dict.json"
        )
        # Post-processing
        cells_pred_dict = postproc(
            images_path, cell_icdar_pred_dict, dataclass="cell",
            datatype="icdar", annotations="", post_path="",
            postprocess="rule", thresh=0.7
        )

        cell_icdar_pred_dict = rename_key(cell_icdar_pred_dict)

        # Build col and row indices from cell predictions
        builder = BuilderRowColIndices()
        cells_pred_dict = builder.build_row_col_indices(cell_icdar_pred_dict)

    if debug:
        save_dict(eval_path, "cells_pred_dict.json", cells_pred_dict)

    return cells_pred_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../datasets/icdar',
                        help="A path to a folder with \
        table_images, sizes_dict.json\
        column_icdar_preds_dict.json, row_icdar_preds_dict.json,\
        content_icdar_preds_dict.json, cell_icdar_preds_dict.json")
    parser.add_argument('--output_path', default='../datasets/icdar',
                        help="A path to save metrics")
    parser.add_argument('--factor', default="1",
                        help="Scaling factor for row detection output")
    parser.add_argument('--spanning', default='',
                        choices=['cell', 'content'],
                        help="Build spanning cells by combining row and column\
                        predictions with content/cell predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Save all intermediate computations")
    parser.add_argument('--how', default='colrow',
                        help="Make predictions from columns and rows: colrow\
                        or make predictions from cell/content")
    args = parser.parse_args()

    input_path = args.input_path
    images_path = os.path.join(input_path, "table_images")
    eval_path = os.path.join(args.output_path, "evaluation")
    os.makedirs(eval_path, exist_ok=True)

    cells_pred_dict = predict_table_structures(
        input_path, eval_path, images_path,
        args.how, args.spanning, args.debug, args.factor
        )

    # Load ground truth for cells
    cells_gt_dict = load_dict(input_path, "icdar_gt_dict.json")

    if args.debug:
        # Draw predicted and ground truth cells
        tables_preds_gt_path = os.path.join(eval_path, "tables_preds_gt")
        os.makedirs(tables_preds_gt_path, exist_ok=True)
        draw_predictions_gt(
            images_path, cells_gt_dict, cells_pred_dict, tables_preds_gt_path
        )

    # Build adjacency list for ground-truth
    generator = IcdarAdjacencyRelationsGenerator()
    table_strs_gt_dict = generator.generate_adj_rel(cells_gt_dict)
    if args.debug:
        save_dict(eval_path, "table_strs_gt_dict.json", table_strs_gt_dict)

    # Match content for predicted cells from ground-truth cell
    cells_pred_dict = build_content_for_preds(
        cells_pred_dict, cells_gt_dict
    )
    if args.debug:
        save_dict(eval_path, "cell_pred_dict_content.json", cells_pred_dict)

    # Build adjacency list for predicted cells
    table_strs_pred_dict = generator.generate_adj_rel(cells_pred_dict)
    save_dict(eval_path, "table_strs_pred_dict.json",
              table_strs_pred_dict)

    # Compute precision, recall, F1-score per table image
    prec_recall_table_dict, _= compare_adj_rel_lists(
        table_strs_gt_dict, table_strs_pred_dict)
    save_dict(eval_path, "prec_recall_f1_table_dict.json",
              prec_recall_table_dict)

    # Compute precision, recall, f1-score per document, average over documents
    prec_recall_f1_doc_dict = compare_docs(
        prec_recall_table_dict)
    save_dict(eval_path, "prec_recall_f1_doc_dict.json",
              prec_recall_f1_doc_dict)


if __name__ == "__main__":
    main()
