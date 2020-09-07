from nms import nms
import cv2
import os
import argparse
from utils.box_utils import intersection_boxes
from utils.box_utils import box_in_box_dataclass, extend_box, overlap_check
from utils.box_utils import convert_list_xyxy_to_xywh
from utils.draw_utils import draw_rectangles
from utils.line_utils import intersection_lines
from utils.file_utils import save_dict, load_dict
from operator import itemgetter


class ResolveNested():

    def resolve_nested(self, boxes_scores, dataclass):
        """
        Sort boxes_scores from biggest to smallest box.
        If a given box contain other boxes and among them there is at least one
        that has higher confidence than the given box delete the given box,
        otherwise delete all nested boxes
        Args:
            boxes_scores: a list of (box, score)
            dataclass: row/column/cell
        Returns:
            boxes_scores: a list of (box, score) that does not contain boxes
                that have another boxes inside them
        """
        # Sort by size: starts from biggest
        boxes_scores.sort(key=lambda x: self._size_box(x[0], dataclass),
                          reverse=True)
        idx_to_del = []
        for idx1, (box1, score1) in enumerate(boxes_scores):
            contains = []  # a list of boxes inside box1
            for idx2, (box2, score2) in enumerate(boxes_scores):
                if box1 == box2:
                    continue
                if box_in_box_dataclass(box2, box1, dataclass):
                    contains.append((idx2, score2))
            if contains:  # find the max score among nested boxes
                max_score = max(contains, key=itemgetter(1))[1]
                if max_score >= score1:
                    idx_to_del.append(idx1)
                else:
                    for (idx, score) in contains:
                        idx_to_del.append(idx)

        idx_to_del = list(set(idx_to_del))
        for i in sorted(idx_to_del, reverse=True):
            del boxes_scores[i]

        return boxes_scores

    def _size_box(self, box, dataclass):
        """
        If dataclass  is column/row, compute width/height of the box
        if dataclass is cell, compute the area of the box
        """
        if dataclass == "column":
            return box[2] - box[0]
        elif dataclass == "row":
            return box[3] - box[1]
        elif dataclass in ["cell", "content"]:
            return (box[3] - box[1]) * (box[2] - box[0])


def find_non_overlap(boxes_scores):
    """
    Split a list into two lists: non-overlapping, overlapping+nested.
    Args:
        boxes_scores: a list of (box,score)
    Returns:
        boxes_scores_post: a list of (box,score) that do not overlap
        boxes_scores: a list of (box,score) that overlap
    """
    boxes_scores_post = []
    idx_to_del = []

    for idx, (box1, score1) in enumerate(boxes_scores):
        overlap = False
        for box2, score2 in boxes_scores:
            if box1 == box2:
                continue
            if intersection_boxes(box1, box2):
                overlap = True
                break
        if not overlap:
            idx_to_del.append(idx)
            boxes_scores_post.append((box1, score1))
    for i in sorted(idx_to_del, reverse=True):
        del boxes_scores[i]

    return boxes_scores_post, boxes_scores


class ResolveOverlaps():

    def __init__(self, dataclass, thresh, boxes_scores):
        self.dataclass = dataclass
        self.thresh = thresh
        self.boxes_scores = boxes_scores

    def resolve_overlaps(self):
        """
        Resolve overlapping boxes:
        1. if self.dataclass is 'cell' then find the direction of bigger
            overlap;
        2. for every pair of overlapping boxes:
           if overlap bigger than threshold keep a box with higher score,
           otherwise, clip the bigger box
        """
        boxes_scores_post = []
        # Sort by confidence score: from lower to higher
        self.boxes_scores.sort(key=lambda x: x[1])
        if self.dataclass in ['column', 'row']:
            dataclass_temp = self.dataclass
        while len(self.boxes_scores):
            box1, score1 = self.boxes_scores.pop(-1)  # the highest score
            for idx, box_score in reversed(list(enumerate(self.boxes_scores))):
                box, score = box_score
                if not intersection_boxes(box, box1):
                    continue
                if self.dataclass in ['column', 'row']:
                    overlap_fraction, line_min, line1_len = \
                        self._compute_overlap_fraction(
                            dataclass_temp, box, box1
                        )
                elif self.dataclass in ['cell', 'content']:
                    overlap_fraction, line_min, line1_len, dataclass_temp =\
                        self._compute_overlap_fraction_cell(box, box1)
                if overlap_fraction >= self.thresh:
                    # keep higher score=> delete box
                    del self.boxes_scores[idx]
                else:
                    # clip bigger one
                    if line_min == line1_len:  # box1 is smaller
                        # clip box
                        if dataclass_temp == "column":
                            if box1[2] < box[2]:  # box1 on the left of box
                                box = [box1[2], box[1], box[2], box[3]]
                            else:  # box1 on the right of box
                                box = [box[0], box[1], box1[0], box[3]]
                        else:
                            if box1[3] < box[3]:  # box1 on the top of box
                                box = [box[0], box1[3], box[2], box[3]]
                            else:  # box1 on the bottom of box
                                box = [box[0], box[1], box[2], box1[1]]
                        self.boxes_scores[idx] = (box, score)
                    else:  # box is smaller
                        # clip box1
                        if dataclass_temp == "column":
                            if box1[0] < box[0]:  # box1 on the left of box
                                box1 = [box1[0], box1[1], box[0], box1[3]]
                            else:  # box1 on the right of box
                                box1 = [box[2], box1[1], box1[2], box1[3]]
                        else:
                            if box1[1] < box[1]:  # box1 on the top of box
                                box1 = [box1[0], box1[1], box1[2], box[1]]
                            else:  # box1 on the bottom of box
                                box1 = [box1[0], box[3], box1[2], box1[3]]

            if not(box1[0] == box1[2] or box1[1] == box1[3]):
                boxes_scores_post.append((box1, score1))

        return boxes_scores_post

    def _compute_overlap_fraction(self, dataclass, box, box1):
        """
        Depending on the dataclass consider boxes as horizontal or vertical
        lines:
        if dataclass is "column"=> horizontal lines,
        if dataclass is "row"=> vertical lines.
        Compute the ratio of overlapping line to the length of
            the shortest line
        Args:
            dataclass: row/column/cell
            box, box1: are defined by their top-left and bottom-right coords
        Returns:
            overlap_fraction: a ratio of overlapping length to the length of
            the shortest line
            line_min: the length of the shortest line
            line1_len: the length of the line associated with box1
        """
        if dataclass == "column":  # horizontal lines
            line = (box[0], box[2])
            line1 = (box1[0], box1[2])
        elif dataclass == "row":  # vertical lines
            line = (box[1], box[3])
            line1 = (box1[1], box1[3])
        intersec = intersection_lines(line, line1)
        line_len = line[1] - line[0]
        line1_len = line1[1] - line1[0]
        line_min = min(line_len, line1_len)
        overlap_fraction = intersec / line_min
        return overlap_fraction, line_min, line1_len

    def _compute_overlap_fraction_cell(self, box, box1):
        """
        Detect in which direction the cells have a bigger overlap and
        then compute _compute_overlap_fraction in that direction
        """
        overlap_horiz, line_min_horiz, line1_len_horiz = \
            self._compute_overlap_fraction("column", box, box1)
        overlap_vert, line_min_vert, line1_len_vert = \
            self._compute_overlap_fraction("row", box, box1)
        if overlap_horiz > overlap_vert:
            overlap_fraction = overlap_vert
            line_min = line_min_vert
            line1_len = line1_len_vert
            dataclass = "row"
        else:
            overlap_fraction = overlap_horiz
            line_min = line_min_horiz
            line1_len = line1_len_horiz
            dataclass = "column"
        return overlap_fraction, line_min, line1_len, dataclass

def box_is_line(boxes_scores):
    idx_to_del = []
    for idx, box_score in enumerate(boxes_scores):
        box, score = box_score
        x1,y1,x2,y2 = box
        if x1==x2 or y1==y2:
            idx_to_del.append(idx)
    for i in sorted(idx_to_del, reverse=True):
        del boxes_scores[i]
    return boxes_scores

def postproc_preds(dataclass, boxes, scores, thresh):
    """
    Post-processing of predicted boxes: resolve overlapping and nested boxes
    Args:
        dataclass: row/column/cell
        boxes: a list of boxes [xyxy]
        scores: a list of scores
        thresh: a threshold of overlapping to decide whether to delete one of
            the overlapping boxes or to keep both but clip one of them
    """
    if not overlap_check(boxes):
        return boxes, scores

    boxes_scores = [(box, score) for box, score in zip(boxes, scores)]
    boxes_scores_non_overlap = []
    boxes_scores = box_is_line(boxes_scores)
    while True:
        # Step 1: keep all non-overlaping boxes, despite of their score
        boxes_scores_post, boxes_scores = find_non_overlap(boxes_scores)
        boxes_scores_non_overlap.extend(boxes_scores_post)

        # Step 2: resolve nested boxes
        resolve_nested = ResolveNested()
        boxes_scores = resolve_nested.resolve_nested(boxes_scores, dataclass)
        # boxes_scores_non_overlap.extend(boxes_scores)
        # break

        # Step 3: keep all non-overlaping boxes, despite of their score
        boxes_scores_post, boxes_scores = find_non_overlap(boxes_scores)
        boxes_scores_non_overlap.extend(boxes_scores_post)

        if len(boxes_scores) == 0:
            break
        # Step 4: resolve ovelapping boxes: clip or choose
        resolve_overlaps = ResolveOverlaps(dataclass, thresh, boxes_scores)
        boxes_scores = resolve_overlaps.resolve_overlaps()

        # Step 5: keep all non-overlaping boxes, despite of their score
        boxes_scores_post, boxes_scores = find_non_overlap(boxes_scores)
        boxes_scores_non_overlap.extend(boxes_scores_post)
        if len(boxes_scores) == 0:
            break

    boxes_post = [box for box, score in boxes_scores_non_overlap]
    scores_post = [score for box, score in boxes_scores_non_overlap]

    return boxes_post, scores_post


def mid_point(boxes, width_height, dataclass):
    """
    Extend boxes:
    increase the width of a column until the midpoint between 2 columns
    increase the height of a row until the midpoint between 2 rows
    """
    if len(boxes) <= 1:
        return boxes
    boxes_wide = []
    if dataclass == "column":
        boxes.sort(key=itemgetter(0))  # sort by x-coord
        box_left = boxes[0]
        box_left[0] = 0
        for box_right in boxes[1:]:
            x1, y1, x2, y2 = box_left
            x3, y3, x4, y4 = box_right
            boxes_wide.append([x1, y1, x2 + int((x3 - x2) / 2), y2])
            box_left = [x2 + int((x3 - x2) / 2), y3, x4, y4]
        boxes_wide.append([x2 + int((x3 - x2) / 2), y3, width_height, y4])
    elif dataclass == "row":
        boxes.sort(key=itemgetter(1))  # sort by y-coord
        box_top = boxes[0]
        box_top[1] = 0
        for box_bottom in boxes[1:]:
            x1, y1, x2, y2 = box_top
            x3, y3, x4, y4 = box_bottom
            boxes_wide.append([x1, y1, x2, y2 + int((y3 - y2) / 2)])
            box_top = [x3, y2 + int((y3 - y2) / 2), x4, y4]
        boxes_wide.append([x3, y2 + int((y3 - y2) / 2), x4, width_height])

    return boxes_wide


def postproc(images_path, predictions_dict, dataclass, datatype, annotations,
             post_path, postprocess, thresh=0.7,factor=1, extend = False):
    """
    Post-process predicted boxes
    Args:
        images_path: a path with table images
        predictions_dict: original output of mask r-cnn
        dataclass: column/row/cell
        datatype: real/unlv/icdar
        annotations: loaded from .json ground-truth
        post_path: path to save table images with post-processed predictions
        postprocess: nms/rule
        thresh: if rule-based postprocess, then choose a threshold for
            resolving overlapping boxes
    """
    if factor != "1":
        # for stretched tables return to the coordinates of the original image
        predictions_dict = postproc_stretched_pred(
            predictions_dict, float(factor)
        )
    for table_name, predictions in predictions_dict.items():
        img_path = os.path.join(images_path, table_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        if annotations:
            if datatype in ["unlv", "icdar", "ctdar"]:
                bbox_gt = annotations[table_name]['cells_list']
            else:
                instances = annotations[table_name]
                bbox_gt = [instance["bbox"] for instance in instances]

        bbox_predictions = predictions["bbox_predictions"]
        scores = predictions["scores"]
        if postprocess == "nms":
            bbox_predictions_xywh = convert_list_xyxy_to_xywh(bbox_predictions)
            best_indices = nms.boxes(
                bbox_predictions_xywh, scores, nms_threshold=0.001
            )
            bbox_predictions = [bbox_predictions[i] for i in best_indices]
            scores = [scores[i] for i in best_indices]

        if postprocess == "rule":
            if datatype in ['icdar', 'unlv', 'ctdar'] or extend:
                if dataclass == "column":
                    bbox_predictions = extend_box(
                        bbox_predictions, height, column=True
                    )
                    if extend:
                        bbox_gt = extend_box(bbox_gt, height, column=True)                        
                elif dataclass == "row":
                    bbox_predictions = extend_box(
                        bbox_predictions, width, column=False
                    )
                    if extend:
                        bbox_gt = extend_box(bbox_gt, width, column=False)
            
            bbox_predictions, scores = postproc_preds(
                dataclass, bbox_predictions, scores, thresh
            )
            
            if datatype == "unlv":
                if dataclass == "column":
                    bbox_predictions = mid_point(
                        bbox_predictions, width, dataclass
                    )
                elif dataclass == "row":
                    bbox_predictions = mid_point(
                        bbox_predictions, height, dataclass
                    )
                    
        if overlap_check(bbox_predictions):  # for debug
            print("OVERLAP")
            break

        predictions["bbox_predictions"] = bbox_predictions
        predictions["scores"] = scores
        if post_path:
            #if annotations:
            #    img = draw_rectangles(img, bbox_gt, (0, 255, 0), 3)
            img = draw_rectangles(img, bbox_predictions, (255, 0, 0), 3)
            table_post_path = os.path.join(post_path, table_name)
            cv2.imwrite(table_post_path, img)
        if annotations and extend:
            for idx, bbox in enumerate(bbox_gt):
                #print("before: ", annotations[table_name][idx]["bbox"])
                annotations[table_name][idx]["bbox"]=bbox
                #print("after: ", bbox)
    
    if annotations and extend:
        return predictions_dict, annotations
    return predictions_dict


def postproc_stretched_pred(predictions_dict, factor):
    """
    Args:
        predictions_dict: predictions are made on vertically stretched images
        factor: a factor that was used for stretching images
    Return:
        predictions_dict: predictions for non-stretched images
    """
    for table_name, predictions in predictions_dict.items():
        bbox_predictions = predictions["bbox_predictions"]
        boxes = []
        for x1, y1, x2, y2 in bbox_predictions:
            boxes.append([x1, int(y1 / factor), x2, int(y2 / factor)])
        predictions_dict[table_name]["bbox_predictions"] = boxes
    return predictions_dict


def main():
    """
    To run post-processing for predicted boxes the following files needed in
    the output folder:
        a file with predictions
        a file with annotations (optional)
        a folder with 'table_images'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="../input",
                        help="A path with predictions and table images")
    parser.add_argument('--output_path', default="../output",
                        help="A path with to save post-processed predictions")
    parser.add_argument('--dataclass', default="column",
                        choices=['column', 'row', 'cell', 'content'],
                        help="Choose column/row/cell/content")
    parser.add_argument('--datatype', default="real",
                        choices=['icdar', 'ctdar', 'unlv', 'real'],
                        help="Choose icdar/ctdar/unlv/real")
    parser.add_argument('--post', default="rule",
                        choices=['rule', 'nms'],
                        help="Choose nms or rule")
    parser.add_argument('--thresh', default="0",
                        help="Threshold for rule based post-processing")
    parser.add_argument('--save', action="store_true",
                        help="Save the post-processed table images")
    parser.add_argument('--factor', default="1",
                        help="Choose vertical stretching factor")

    args = parser.parse_args()

    thresh_str = "".join(str(args.thresh).split("."))
    input_path = args.input_path
    output_path = args.output_path
    datatype = args.datatype
    dataclass = args.dataclass
    postproc_path = os.path.join(
        output_path, datatype + "_" + dataclass + "_" +
        args.post + "_" + thresh_str + "_postproc"
        )
    os.makedirs(postproc_path, exist_ok=True)

    if datatype == "real":
        images_path = os.path.join(input_path, "test_table_images")
    elif datatype in ['icdar', 'ctdar', 'unlv']:
        images_path = os.path.join(input_path, "table_images")
    predictions_dict = load_dict(
        input_path, dataclass + "_" + datatype + "_preds_dict.json"
    )

    annotations = ""
    if datatype in ["unlv", "icdar", "ctdar"]:
        annotations = load_dict(input_path, datatype + "_gt_dict.json")
    elif datatype == "real":
        annotations_path = os.path.join(input_path,
                                        "test_" + dataclass + "_annotations")
        names = os.listdir(annotations_path)
        annotations = {}
        for name in names:
            annotation = load_dict(annotations_path, name)
            annotations.update(annotation)

    post_pred_path = ""
    if args.save:
        post_pred_path = os.path.join(
            postproc_path, datatype + "_" + dataclass + "_" +
            args.post + "_" + thresh_str + "_pred"
        )
        os.makedirs(post_pred_path, exist_ok=True)

    #if args.factor != "1":
        # for stretched tables return to the coordinates of the original image
    #    predictions_dict = postproc_stretched_pred(
    #        predictions_dict, float(args.factor)
    #    )
    preds_dict_post = postproc(images_path, predictions_dict, dataclass,
                               datatype, annotations, post_pred_path,
                               args.post, float(args.thresh), args.factor)

    save_dict(
        postproc_path, datatype + "_" + dataclass + "_" +
        args.post + "_" + thresh_str + "_pred_dict.json", preds_dict_post
    )


if __name__ == "__main__":
    main()
