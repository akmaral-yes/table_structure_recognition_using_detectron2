import argparse
import os
from utils.file_utils import save_dict, load_dict
from utils.draw_utils import draw_preds_annotations
from metrics.segments_detection_metrics import SegmentsDetectionMetrics
from metrics.mean_avg_prec import COCOmAP
from detectron2_tables.postproc_maskrcnn import postproc


def main():
    """
    Run the code: python measures.py --measure segments
    --input_path ../../8_unlv_dataset/output --datatype unlv --dataclass row
    --draw --post rule --thresh 0.6

    Depending on a datatype not all metrics(COCOmAP, segments detecion metrics)
    can be computed because of the absence of ground truth
        real:
            COCOmAP for row/column/cell/content
            segments detecion metrics for row/column
        unlv:
            COCOmAP for row/column/content position
            segments detecion metrics for row/column
        icdar:
            COCOmAP for content position
        ctdar:
            COCOmAP for content position

    To compute measures the following files are needed:
    - Mask R-CNN output predictions
    - table images, if post-processing is needed
    - ground truth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../datasets/unlv',
                        help="A path to a folder with Mask R-CNN predictions\
                        (ex.: row_unlv_preds_dict.json), table images,\
                        ground truth")
    parser.add_argument('--output_path', default='../datasets/unlv',
                        help="A path to save metrics")
    parser.add_argument('--measure', default="segments",
                        choices=['map', 'segments'],
                        help="Choose which metric to compute")
    parser.add_argument('--post', default="",
                        choices=['nms', 'rule'],
                        help="Choose post-processing type")
    parser.add_argument('--extend', action='store_true',
                        help="Extend column/row to the image height/width \
                        in predictions and ground truth")
    parser.add_argument('--thresh', default="0",
                        help="Threshold for rule based post-processing")
    parser.add_argument('--datatype', default="real",
                        #choices=['icdar', 'unlv', 'ctdar', 'real'],
                        help="Which data to use")
    parser.add_argument('--dataclass', default="column",
                        choices=['column', 'row', 'cell', 'content'],
                        help="Choose dataclass")
    parser.add_argument('--thresh_correct', default="0.9",
                        help="Choose threshold for segments metric (0, 1)")
    parser.add_argument('--draw', action='store_true',
                        help="Draw predictions and ground truth")
    parser.add_argument('--factor', default="1",
                        help="Choose vertical stretching factor")

    args = parser.parse_args()
    dataclass = args.dataclass
    datatype = args.datatype
    thresh_str = "".join(str(args.thresh).split("."))
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    if args.extend:
        eval_path = os.path.join(
            output_path, args.measure + "_" + datatype + "_" + dataclass + "_"
            + args.post + "_" + thresh_str + "_" + "extend_eval"
            )
    else:
        eval_path = os.path.join(
            output_path, args.measure + "_" + datatype + "_" + dataclass + "_"
            + args.post + "_" + thresh_str + "_eval"
            )       
    os.makedirs(eval_path, exist_ok=True)

    # Load predictions made by Mask R-CNN
    preds_dict = load_dict(
        input_path, dataclass + "_" + datatype + "_preds_dict.json"
    )

    if args.post or args.draw:
        if datatype in ["real", "real3"]:
            images_path = os.path.join(input_path, "test_table_images")
        elif datatype in ['icdar', 'ctdar', 'unlv']:
            images_path = os.path.join(input_path, "table_images")


    # Load ground truth/annotations
    if datatype in ["unlv", "icdar", "ctdar"]:
        annotations = load_dict(input_path, datatype + "_gt_dict.json")
    elif datatype in ["real", "real3"]:
        annotations_path = os.path.join(input_path,
                                        "test_" + dataclass + "_annotations")
        names = os.listdir(annotations_path)
        annotations = {}
        for name in names:
            annotation = load_dict(annotations_path, name)
            annotations.update(annotation)

    # Apply post-processing if required
    if args.post:
        if args.extend: 
            preds_dict, annotations = postproc(
                images_path, preds_dict, dataclass, datatype, annotations, "",
                postprocess = args.post, thresh = float(args.thresh),
                factor = args.factor, extend = args.extend
            )
        else:
            preds_dict = postproc(
                images_path, preds_dict, dataclass, datatype, annotations, "",
                postprocess = args.post, thresh = float(args.thresh),
                factor = args.factor
            )

    if args.draw:
        tables_preds_gt_path = os.path.join(eval_path, "tables_preds_gt")
        os.makedirs(tables_preds_gt_path, exist_ok=True)
        draw_preds_annotations(images_path, annotations, preds_dict,
                               tables_preds_gt_path, datatype)

    measures_path = os.path.join(
        eval_path, datatype + '_' + args.measure +
        '_' + dataclass + "_" + args.post + "_" + thresh_str + ".csv"
        )

    if args.measure == "segments":
        segments_det_metrics = SegmentsDetectionMetrics()
        measures_dict, measures_df = \
            segments_det_metrics.compute_measures_batch(
                images_path, preds_dict, annotations, args.datatype,
                dataclass, float(args.thresh_correct))
        save_dict(eval_path, dataclass + "_" + args.post + "_" + thresh_str +
                  "_measures.json", measures_dict)
        measures_df.to_csv(measures_path, index=False)

    if args.measure == "map":
        coco_map = COCOmAP()
        measures = coco_map.mean_avg_prec_batch(
            preds_dict, annotations, datatype, dataclass
        )
        measures.to_csv(measures_path, index=False)


if __name__ == "__main__":
    main()
