import math


def compare_adj_rel_lists(gt_table_structures, pred_table_structures):
    """
    Given lists of ground-truth and predicted adjacency relationships for a
    batch of tables, compute true positives, false positives,
    # of preds, # of gt, precision, recall and F1-score per table.
    Compute average F1-score over the given batch of tables
    """
    metrics_dict = {}
    f1_score_accum = 0
    tables_num = len(gt_table_structures.keys())
    for table_name, gt_structure in gt_table_structures.items():
        pred_structure = pred_table_structures[table_name]
        tp = 0
        fp = 0
        preds = len(pred_structure)
        gts = len(gt_structure)
        for line_pred in pred_structure:
            if line_pred in gt_structure:
                tp += 1
            else:
                fp += 1
        metrics_dict[table_name] = {}
        precision = tp / preds
        recall = tp / gts
        metrics_dict[table_name]["tp"] = tp
        metrics_dict[table_name]["preds"] = preds
        metrics_dict[table_name]["gts"] = gts
        metrics_dict[table_name]["precision"] = precision
        metrics_dict[table_name]["recall"] = recall
        if precision + recall == 0:
            metrics_dict[table_name]["F1-score"] = 0
            print("precision +recall=0: ", table_name)
        else:
            metrics_dict[table_name]["F1-score"] = \
                2 * precision * recall / (precision + recall)
        f1_score_accum += metrics_dict[table_name]["F1-score"]
    f1_score = f1_score_accum / tables_num

    f1_sq_dev_sum = 0
    for table_name in metrics_dict.keys():
        f1_sq_dev_sum = f1_sq_dev_sum + \
            (metrics_dict[table_name]["F1-score"] - f1_score) ** 2

    metrics_dict["total"] = {}
    metrics_dict["total"]["F1-score"] = f1_score
    metrics_dict["total"]["F1-score-se"] = \
        round(math.sqrt(f1_sq_dev_sum/(tables_num - 1))/math.sqrt(tables_num), 4)

    
    return metrics_dict, f1_score


def compare_docs(table_dict):
    """
    Compute true positives, # of preds, # of gt, precision, recall and
    F1-score per document and average precision, recall, f1-score
    Args:
        table_dict: a dictionary: key - table_name, values:
                    "tp" - number of true positives adj relationships
                    "preds" - number of predictions adj relationships
                    "gts" - number of grount-truth adj relathionships
                    "precision" - precision per table
                    "recall" - recall per table
                    "F1-score" - F1-score per table
    Returns:
        doc_dict: the same as table_dict but per document
    """
    doc_dict = {}
    # Accumulate tp, preds, gts per document
    for table_name in table_dict.keys():
        doc_name = table_name.split("_")[0]
        if doc_name == "total":
            continue
        if doc_name not in doc_dict.keys():
            doc_dict[doc_name] = {}
            doc_dict[doc_name]["tp"] = table_dict[table_name]["tp"]
            doc_dict[doc_name]["preds"] = table_dict[table_name]["preds"]
            doc_dict[doc_name]["gts"] = table_dict[table_name]["gts"]
        else:
            doc_dict[doc_name]["tp"] += table_dict[table_name]["tp"]
            doc_dict[doc_name]["preds"] += table_dict[table_name]["preds"]
            doc_dict[doc_name]["gts"] += table_dict[table_name]["gts"]

    # Compute precision, recall, F1-score per document
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0

    for doc_name in doc_dict.keys():
        precision = doc_dict[doc_name]["tp"]/doc_dict[doc_name]["preds"]
        doc_dict[doc_name]["precision"] = precision
        precision_sum += precision

        recall = doc_dict[doc_name]["tp"]/doc_dict[doc_name]["gts"]
        doc_dict[doc_name]["recall"] = recall
        recall_sum += recall

        f1 = 2*precision*recall/(precision+recall)
        doc_dict[doc_name]["F1-score"] = f1
        f1_sum += f1

    # Compute precision, recall, F1-score for the whole set of documents
    number_docs = len(doc_dict)
    doc_dict["total"] = {}
    doc_dict["total"]["precision"] = round(precision_sum/number_docs,4)
    doc_dict["total"]["recall"] = round(recall_sum/number_docs, 4)
    doc_dict["total"]["F1-score"] = round(f1_sum/number_docs, 4)

    precision_sq_dev_sum = 0
    recall_sq_dev_sum = 0
    f1_sq_dev_sum = 0
    for doc_name in doc_dict.keys():
        precision_sq_dev_sum = precision_sq_dev_sum + \
            (doc_dict[doc_name]["precision"] - doc_dict["total"]["precision"]) ** 2
        recall_sq_dev_sum = recall_sq_dev_sum + \
            (doc_dict[doc_name]["recall"] - doc_dict["total"]["recall"]) ** 2
        f1_sq_dev_sum = f1_sq_dev_sum + \
            (doc_dict[doc_name]["F1-score"] - doc_dict["total"]["F1-score"]) ** 2

    doc_dict["total"]["precision-se"] = round(
        math.sqrt(precision_sq_dev_sum/(number_docs-1))/math.sqrt(number_docs), 4
        )
    doc_dict["total"]["recall-se"] = round(math.sqrt(recall_sq_dev_sum/(number_docs - 1))/math.sqrt(number_docs), 4)
    doc_dict["total"]["F1-score-se"] = round(math.sqrt(f1_sq_dev_sum/(number_docs - 1))/math.sqrt(number_docs), 4)

    return doc_dict