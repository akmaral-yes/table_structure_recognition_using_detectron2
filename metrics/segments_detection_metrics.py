import cv2
import os
import pandas as pd
from utils.box_utils import intersection_boxes, area_computation


class SegmentsDetectionMetrics():

    def compute_measures_batch(
            self, images_path, predictions_dict, annotations, datatype,
            dataclass, thresh_correct=0.9
    ):
        """
        Computes 6 segments detection metrics for the batch of table images
        """
        columns = ["Measures", "Number"]
        measures_df = pd.DataFrame()
        measures_dict = {}
        object_num = 0
        pred_num = 0
        measures_total = [0, 0, 0, 0, 0, 0]

        for table_name, prediction in predictions_dict.items():
            print("table_name: ", table_name)
            img_path = os.path.join(images_path, table_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            if datatype == "unlv":
                bbox_gt = annotations[table_name][dataclass]
            else:
                instances = annotations[table_name]
                bbox_gt = [instance["bbox"] for instance in instances]
            #bbox_gt = [[0,y1,width,y2] for x1,y1,x2,y2 in bbox_gt]

            bbox_predictions = prediction["bbox_predictions"]
            #bbox_predictions = [[0,y1,width,y2] for x1,y1,x2,y2 in bbox_predictions]
            scores = prediction["scores"]
            measures = self._compute_measures(
                bbox_gt, bbox_predictions, thresh_correct)
            measures_dict[table_name] = {}
            measures_dict[table_name]["measures"] = measures
            measures_dict[table_name]["gt"] = bbox_gt
            measures_dict[table_name]["predictions"] = bbox_predictions
            measures_dict[table_name]["scores"] = scores
            object_num += len(bbox_gt)
            pred_num += len(bbox_predictions)
            measures_total = [sum(x) for x in zip(measures_total, measures)]

        measures_df = measures_df.append(pd.DataFrame(
            [["correct_detections", measures_total[0]]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["partial_detections", measures_total[1]]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["over_segmentation", measures_total[2]]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["under_segmentation", measures_total[3]]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["missed_segments", measures_total[4]]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["fp_detections", measures_total[5]]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["gt_num", object_num]], columns=columns)
            )
        measures_df = measures_df.append(pd.DataFrame(
            [["pred_num", pred_num]], columns=columns)
            )

        return measures_dict, measures_df

    def _compute_measures(self, bbox_gt, bbox_predictions, thresh_correct=0.9):
        """
        Compute correct_detections, partial_detections, over_segmentation,
        under_segmentation, missed_segments, fp_detections for 1 table
        Args:
            bbox_gt: a list of ground_truth bboxes
            bbox_predictions: a list of predicted bboxes
            thresh_correct: a threshold of overlapping to classify a box as
                correct
        """
        correct_detections, partial_detections = 0, 0
        over_segmentation, under_segmentation = 0, 0
        missed_segments, fp_detections = 0, 0

        for bbox_predicted in bbox_predictions:
            area_bbox_predicted = area_computation(bbox_predicted)
            intersection_over_gt_list = []
            intersection_over_predicted_list = []
            #print("bbox_predicted: ",bbox_predicted)
            for bbox in bbox_gt:
                intersection = intersection_boxes(bbox, bbox_predicted)
                area_intersection = area_computation(intersection)
                area_bbox = area_computation(bbox)
                intersection_over_gt = area_intersection / area_bbox
                intersection_over_gt_list.append(intersection_over_gt)
                intersection_over_predicted = \
                    area_intersection / area_bbox_predicted
                intersection_over_predicted_list.append(
                    intersection_over_predicted)

            # False positive detections
            if all(x < (1 - thresh_correct)
                   for x in intersection_over_predicted_list):
                fp_detections += 1
            # Correctly detected or partially detected
            overlaps_gt = 0
            for idx, intersection_over_gt in enumerate(
                    intersection_over_gt_list):
                if intersection_over_gt >= thresh_correct:
                    temp_list = intersection_over_predicted_list.copy()
                    del temp_list[idx]
                    if all(x < (1 - thresh_correct) for x in temp_list):
                        correct_detections += 1
                elif (intersection_over_gt > 0.1):
                    temp_list = intersection_over_predicted_list.copy()
                    del temp_list[idx]
                    if all(x < (1 - thresh_correct) for x in temp_list):
                        partial_detections += 1
                if intersection_over_predicted_list[idx] < thresh_correct and \
                        intersection_over_predicted_list[idx] > \
                        (1 - thresh_correct):
                    overlaps_gt += 1

            # Under segmentation
            if overlaps_gt >= 2:
                under_segmentation += 1

        # Over segmentation
        for bbox in bbox_gt:
            area_bbox = area_computation(bbox)
            overlaps_predictions = 0
            intersection_over_gt_list = []

            for bbox_predicted in bbox_predictions:
                area_bbox_predicted = area_computation(bbox_predicted)
                intersection = intersection_boxes(bbox, bbox_predicted)
                area_intersection = area_computation(intersection)

                intersection_over_gt = area_intersection / area_bbox
                intersection_over_gt_list.append(intersection_over_gt)
                if intersection_over_gt < thresh_correct and \
                        intersection_over_gt > (1 - thresh_correct):
                    overlaps_predictions += 1
                intersection_over_predicted = \
                    area_intersection / area_bbox_predicted

            # Missed segments
            if all(x < (1 - thresh_correct)
                   for x in intersection_over_gt_list):

                missed_segments += 1
            if overlaps_predictions >= 2:
                over_segmentation += 1

        return correct_detections, partial_detections, over_segmentation, \
            under_segmentation, missed_segments, fp_detections
