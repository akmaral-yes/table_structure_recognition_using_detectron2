from operator import itemgetter
import pandas as pd
from utils.box_utils import compute_iou
import multiprocessing
import functools


class COCOmAP():

    def mean_avg_prec_batch(
        self, predictions_dict, annotations, datatype, dataclass
    ):
        """
        Compute COCO mAP(mean Average Precision): average over AP
        ([.50:.05:.95])
        """
        col_measures = ["thresh", "AP"]
        measures = pd.DataFrame()
        thresh_list = [x / 100 for x in range(50, 100, 5)]
        ap_sum = 0
        processes_number = multiprocessing.cpu_count()
        p = multiprocessing.Pool(processes_number)
        self.annotations = annotations
        self.datatype = datatype
        self.dataclass = dataclass
        self.predictions_dict = predictions_dict
        with p as pool:
            worker = functools.partial(self._avg_prec)
            for thresh, ap in zip(
                thresh_list, pool.imap_unordered(
                    worker, thresh_list)):
                measures = measures.append(pd.DataFrame(
                    [[thresh, ap]], columns=col_measures))
                ap_sum += ap
            pool.close()
            pool.join()
        measures.sort_values('thresh')
        print("mAP: ", ap_sum / 10)
        measures = measures.append(pd.DataFrame(
            [["mean", ap_sum / 10]], columns=col_measures))
        return measures

    def _avg_prec(self, thresh):
        stats = pd.DataFrame()
        num_gt = 0
        num_preds = 0
        for table_name, prediction in self.predictions_dict.items():
            # print("table_name: ", table_name)
            if self.datatype in ['real', 'real3']:
                instances = self.annotations[table_name]
                bbox_gt = [instance["bbox"] for instance in instances]
            elif self.datatype in ['icdar', 'ctdar', 'unlv']:
                if self.dataclass == 'content':
                    bbox_gt = self.annotations[table_name]['cells_list']
                elif self.dataclass in ['row', 'column']:
                    bbox_gt = self.annotations[table_name][self.dataclass]
            if len(bbox_gt) == 0:
                # Skip the images without any ground truth
                # unlv 9500_034.xml - no cells bug
                continue
            bbox_predictions = prediction["bbox_predictions"]
            scores = prediction["scores"]
            num_gt += len(bbox_gt)
            num_preds += len(bbox_predictions)
            stats_add = self._compute_statistics(
                bbox_predictions, scores, bbox_gt, thresh)
            stats = stats.append(stats_add, ignore_index=True)
        print("thresh: ", thresh)
        print("num_gt: ", num_gt)
        print("num_preds: ", num_preds)
        ap = self._compute_ap(stats, num_gt, num_preds)

        return ap

    def _compute_statistics(self, bbox_predictions, scores, bbox_gt, thresh):
        """
        Returns:
            pandas dataframe with the following columns:
            ["pred_box", "conf", "tp"]
        """
        columns = ["pred_box", "conf", "tp"]
        stats = pd.DataFrame(columns=columns)
        tp = 0
        for (bbox_prediction, score) in zip(bbox_predictions, scores):
            iou_with_gt = []
            tp = 0
            for idx, bbox in enumerate(bbox_gt):
                iou = compute_iou(bbox_prediction, bbox)
                iou_with_gt.append((iou, idx))
            iou_with_gt.sort(key=itemgetter(0), reverse=True)
            iou_max, idx_max = iou_with_gt[0]
            if iou_max >= thresh:
                # Several bbox_prediction overlap with the same ground truth 
                # and these overlaps are the biggest for these bbox_predictions
                # Only the bbox_prediction that has the biggest overlap with
                # the given ground truth is tp, all others are false negative
                bbox_gt_max = bbox_gt[idx_max]
                iou_with_pred = []
                for bbox_prediction1 in bbox_predictions:
                    iou1 = compute_iou(bbox_prediction1, bbox_gt_max)
                    iou_with_pred.append((iou1,bbox_prediction1))
                iou_with_pred.sort(key=itemgetter(0), reverse=True)
                if iou_with_pred[0][1] == bbox_prediction:
                    tp = 1
            stats = stats.append(
                pd.DataFrame([[bbox_prediction, score, tp]], columns=columns)
                )
        return stats

    def _compute_ap(self, stats, num_gt, num_preds):
        """
        Compute average precision:
            1. true positive
            2. true positive accumulated
            3. recall
            4. precision
            5. precision interpolated
            6. area under precsion-recall curve
        """
        stats = stats.sort_values("conf", ascending=False)
        stats["tp_acc"] = stats["tp"].cumsum()
        stats["recall"] = stats["tp_acc"] / num_gt
        stats.reset_index(drop=True, inplace=True)
        stats["precision"] = stats["tp_acc"] / (stats.index + 1)
        stats.loc[0, "prec_int"] = max(stats.loc[:, "precision"])
        prec_int_prev = stats.loc[0, "prec_int"]
        recall_prev = stats.loc[0, "recall"]
        pr = 0  # precision-recall curve
        for i in stats.index[1:]:
            stats.loc[i, "prec_int"] = max(stats.loc[i:, "precision"])
            if prec_int_prev != stats.loc[i, "prec_int"]:
                pr += prec_int_prev * \
                    (stats.loc[i - 1, "recall"] - recall_prev)
                prec_int_prev = stats.loc[i, "prec_int"]
                recall_prev = stats.loc[i, "recall"]
        pr += prec_int_prev * (stats.loc[i, "recall"] - recall_prev)
        print("ap: ", pr)

        return pr
