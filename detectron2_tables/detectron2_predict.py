import cv2
import random
import os
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class Predict():

    def predict_boxes(self, cfg, table_dicts, table_metadata, pred_path, n=0):
        """
        Build predtion boxes for columns/rows/cells
        Args:
            cfg: the trained model config
            table_dicts: a dictionary with annotations
            table_metadata: registered metadata
            pred_path: a path to save drawn predictions, if "" - do not save
            n: number of randomly choosen tables, if 0 - all tables
        Returns:
            a dictionary of table_names and a list of predicted boxes
        """
        predictor = DefaultPredictor(cfg)
        if n != 0:
            table_dicts = random.sample(table_dicts, n)
        predictions = {}
        for table_dict in table_dicts:
            file_path = table_dict["file_name"]
            print("file_path: ", file_path)
            img = cv2.imread(file_path)
            outputs = predictor(img)
            table_name = file_path.split("/")[-1]
            if pred_path:
                v = Visualizer(img[:, :, ::-1],
                               metadata=table_metadata,
                               scale=1,
                               # remove the colors of unsegmented pixels
                               instance_mode=ColorMode.IMAGE_BW
                               )
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                predicted_img_path = os.path.join(pred_path, table_name)
                cv2.imwrite(predicted_img_path, v.get_image()[:, :, ::-1])
            scores_tensor = outputs["instances"].scores.to("cpu")
            scores = scores_tensor.tolist()
            predictions_tensor = outputs["instances"].pred_boxes.to("cpu")
            bbox_predictions = self._convert_to_list(predictions_tensor)
            predictions[table_name] = {}
            predictions[table_name]["bbox_predictions"] = bbox_predictions
            predictions[table_name]["scores"] = scores

        return predictions

    def _convert_to_list(self, bboxes_tensor):
        """
        Convert tensor to a list
        """
        bboxes = []
        for bbox in bboxes_tensor:
            bbox_float = bbox.tolist()
            bbox_int = [round(i) for i in bbox_float]
            bboxes.append(bbox_int)
        return bboxes
