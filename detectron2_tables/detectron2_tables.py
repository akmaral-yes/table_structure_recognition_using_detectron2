import argparse
import cv2
import random
import os
from utils.file_utils import save_dict
from generate_annotations import get_table_dicts, generate_empty_table_dicts
from detectron2_train import do_train
from detectron2_predict import Predict
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer


def visualize_boxes(table_dicts, table_metadata, visualizer_path, n=0):
    """
    Visualize  column/row boxes on table images

    Args:
        n: number of randomly choosen tables, if 0 - all tables
    """
    if n != 0:
        table_dicts = random.sample(table_dicts, n)

    for table_dict in table_dicts:
        file_path = table_dict["file_name"]
        print("file_path: ", file_path)
        img = cv2.imread(file_path)
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=table_metadata, scale=0.3
        )
        vis = visualizer.draw_dataset_dict(table_dict)
        table_name = file_path.split("/")[-1]
        img = vis.get_image()[:, :, ::-1]
        visualized_img_path = os.path.join(visualizer_path, table_name)
        cv2.imwrite(visualized_img_path, img)


def main():
    # python detectron2_tables.py --dataclass row --train
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../datasets',
                        help="A path to folders with table images \
                            (mode + '_tables') and ground-truth dicts \
                            (mode+_gt_tables_dict)")
    parser.add_argument('--output_path', default='../datasets/real',
                        help="A path to save predictions")
    parser.add_argument('--models_path', default='../models',
                        help="A path to save trained models and losses")
    parser.add_argument('--images_path',
                        default='../datasets/ctdar/table_images',
                        help="A path to images for making predictions")
    parser.add_argument('--dataclass', default='column',
                        choices=['column', 'row', 'cell', 'content'],
                        help="Choose column/row/cell/content")
    parser.add_argument('--datatype', default='real4',
                        help="Choose real3/real4/icdar/unlv/ctdar")
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize ground_truth")
    parser.add_argument('--train', action='store_true',
                        help="Train")
    parser.add_argument('--predict', action='store_true',
                        help="Predict")
    parser.add_argument('--save', action='store_true',
                        help="Save images with predictions")
    parser.add_argument('--weights', default='final',
                        help="The number of iterations of the model or final")
    parser.add_argument('--score_thresh', default='0.05',
                        help="A score threshold for predictions")
    parser.add_argument(
        '--loss_period',
        default=10000,
        help="Number of iterations for computing validattion loss")
    parser.add_argument(
        '--checkpoint_period',
        default=10000,
        help="Number of iterations for computing saving the model")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # Register dataset
    if args.datatype in ['icdar', 'unlv', 'ctdar', 'other']:
        DatasetCatalog.register(
            "table_" + args.datatype,
            lambda datatype=args.datatype: generate_empty_table_dicts(
                args.images_path))
        MetadataCatalog.get("table_" + args.datatype).\
            set(thing_classes=[args.dataclass])
    else:
        for mode in ["train", "test"]:
            DatasetCatalog.register(
                "table_" + mode, lambda mode=mode: get_table_dicts(
                    mode, args.datatype, args.dataclass, input_path
                )
            )
            MetadataCatalog.get("table_" + mode).\
                set(thing_classes=[args.dataclass]
                    )

    # Load the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (column/row/cell)

    # Create a folder for saving trained models and losses
    os.makedirs(args.models_path, exist_ok=True)
    cfg.OUTPUT_DIR = os.path.join(args.models_path, args.dataclass)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.visualize:
        # Visualize ground_truth on train dataset
        table_metadata = MetadataCatalog.get("table_train")
        table_dicts = get_table_dicts(
            "train", args.datatype, args.dataclass, input_path
            )
        visualizer_path = os.path.join(output_path, "visualizer")
        os.makedirs(visualizer_path, exist_ok=True)
        visualize_boxes(table_dicts, table_metadata, visualizer_path, n=20)

    if args.train:
        if args.weights != 'final':
            weights_path = os.path.join(
                cfg.OUTPUT_DIR, "model_" + args.weights.zfill(7) + ".pth")
            if os.path.exists(weights_path):
                cfg.MODEL.WEIGHTS = weights_path
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 400000
        cfg.SOLVER.CHECKPOINT_PERIOD = int(args.checkpoint_period)
        # Faster, and good enough for this dataset (default: 512)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        # Run model on validation set with this periodicity
        cfg.TEST.EVAL_PERIOD = int(args.loss_period)
        cfg.DATASETS.TRAIN = ("table_train",)
        cfg.DATASETS.TEST = ("table_test", )
        model = build_model(cfg)
        do_train(cfg, model)

    # COMPUTE PREDICTIONS
    if args.predict:
        if args.weights != 'final':
            weights_path = os.path.join(
                cfg.OUTPUT_DIR, "model_" + args.weights.zfill(7) + ".pth")
            if os.path.exists(weights_path):
                cfg.MODEL.WEIGHTS = weights_path
        print("Prediction started")
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        # Set the testing threshold for this model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.score_thresh)
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000
        if args.datatype in ['icdar', 'unlv', 'ctdar', 'other']:
            cfg.DATASETS.TEST = ("table_" + args.datatype, )
            table_metadata = MetadataCatalog.get("table_" + args.datatype)
            table_dicts = generate_empty_table_dicts(args.images_path)
        else:
            cfg.DATASETS.TEST = ("table_test", )
            table_metadata = MetadataCatalog.get("table_test")
            table_dicts = get_table_dicts(
                "test", args.datatype, args.dataclass, input_path
            )
        if args.save:
            pred_path = os.path.join(
                output_path, args.dataclass + "_" +
                args.datatype + "_pred_images")
            os.makedirs(pred_path, exist_ok=True)
        else:
            pred_path = ""
        predict = Predict()
        preds_dict = predict.predict_boxes(
            cfg, table_dicts, table_metadata, pred_path
        )
        save_dict(output_path, args.dataclass + "_" + args.datatype +
                  "_preds_dict.json", preds_dict)


if __name__ == "__main__":
    main()
