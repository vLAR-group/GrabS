# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import sys
from copy import deepcopy
from uuid import uuid4
from matplotlib import pyplot as plt
import os.path as osp
import numpy as np
import benchmark.util as util
import benchmark.util_3d as util_3d


# ---------- Label info ---------- #
CLASS_LABELS = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
CHAIR_ID = 9
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt["min_region_sizes"] = np.array([100])  # 100 for s3dis, scannet
# distance thresholds [m]
opt["distance_threshes"] = np.array([float("inf")])
# distance confidences
opt["distance_confs"] = np.array([-float("inf")])


def evaluate_matches(use_label, matches, prcurv_save_dir):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # ###########################################
    # global CLASS_LABELS
    # if not use_label:
    #     eval_class_labels = ["class_agnostic"]
    # else:
    #     eval_class_labels = CLASS_LABELS
    # global eval_class_labels
    # ##########################################3

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(eval_class_labels), len(overlaps)), float)
    rc = np.zeros((len(dist_threshes), len(eval_class_labels), len(overlaps)), float)
    pr = np.zeros((len(dist_threshes), len(eval_class_labels), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    for label_name in eval_class_labels:
                        for p in matches[m]["pred"][label_name]:
                            if "uuid" in p:
                                pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(eval_class_labels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["instance_id"] >= 0
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # greedy assignments
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                # group?
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                # small ground truth instances
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = (
                                float(num_ignore) / pred["vert_count"]
                            )
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = (
                        y_true_sorted_cumsum[-1]
                        if len(y_true_sorted_cumsum) > 0
                        else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # recall is the first point on recall curve
                    rc_current = recall[0]
                    pr_current = precision[0]

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # plot and save
                    fig = plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(recall, precision)
                    plt.plot(recall, precision, "r*")
                    plt.grid()
                    plt.xlabel("Recall")
                    plt.xlim((0.0, 1.0))
                    plt.ylabel("Precision")
                    plt.ylim((0.0, 1.0))
                    plt.title(f"PR di={di} iou={overlap_th:.3f} {label_name}")

                    plt.subplot(1, 3, 2)
                    plt.plot(thresholds, precision[:-1])
                    plt.plot(thresholds, precision[:-1], "r*")
                    plt.grid()
                    plt.xlabel("conf TH")
                    plt.xlim((0.0, 1.0))
                    plt.ylabel("Precision")
                    plt.ylim((0.0, 1.0))
                    plt.title(f"P-TH di={di} iou={overlap_th:.3f} {label_name}")

                    plt.subplot(1, 3, 3)
                    plt.plot(thresholds, recall[:-1])
                    plt.plot(thresholds, recall[:-1], "r*")
                    plt.grid()
                    plt.xlabel("conf TH")
                    plt.xlim((0.0, 1.0))
                    plt.ylabel("Recall")
                    plt.ylim((0.0, 1.0))
                    plt.title(f"R-TH di={di} iou={overlap_th:.3f} {label_name}")

                    if prcurv_save_dir is not None:
                        plt.savefig(osp.join(prcurv_save_dir, f"{di}_iou={overlap_th:.3f}_{label_name}.png"))
                    # np.savez_compressed(osp.join(prcurv_save_dir, f"{di}_iou={iou_th:.3f}_{label_name}.npz"),
                    #     precision=precision,
                    #     recall=recall,
                    #     thresholds=thresholds)
                    plt.close()

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                    rc_current = 0.0
                    pr_current = 0.0
                else:
                    ap_current = float("nan")
                    rc_current = float("nan")
                    pr_current = float("nan")
                ap[di, li, oi] = ap_current
                rc[di, li, oi] = rc_current
                pr[di, li, oi] = pr_current

    return ap, rc, pr


def compute_averages(use_label, aps, rcs, prs):
    # ###########################################
    # global CLASS_LABELS
    # if not use_label:
    #     eval_class_labels = ["class_agnostic"]
    # else:
    #     eval_class_labels = CLASS_LABELS
    # global eval_class_labels
    # ##########################################
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["all_rc"] = np.nanmean(rcs[d_inf, :, oAllBut25])
    avg_dict["all_rc_50%"] = np.nanmean(rcs[d_inf, :, o50])
    avg_dict["all_rc_25%"] = np.nanmean(rcs[d_inf, :, o25])
    avg_dict["all_pr"] = np.nanmean(prs[d_inf, :, oAllBut25])
    avg_dict["all_pr_50%"] = np.nanmean(prs[d_inf, :, o50])
    avg_dict["all_pr_25%"] = np.nanmean(prs[d_inf, :, o25])
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(eval_class_labels):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
        avg_dict["classes"][label_name]["rc"] = np.average(rcs[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["rc50%"] = np.average(rcs[d_inf, li, o50])
        avg_dict["classes"][label_name]["rc25%"] = np.average(rcs[d_inf, li, o25])
        avg_dict["classes"][label_name]["pr"] = np.average(prs[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["pr50%"] = np.average(prs[d_inf, li, o50])
        avg_dict["classes"][label_name]["pr25%"] = np.average(prs[d_inf, li, o25])
    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (
        pred["pred_classes"].shape[0]
        == pred["pred_scores"].shape[0]
        == pred["pred_masks"].shape[1])
    for i in range(len(pred["pred_classes"])):
        info = {}
        info["label_id"] = pred["pred_classes"][i]
        info["conf"] = pred["pred_scores"][i]
        info["mask"] = pred["pred_masks"][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def assign_instances_for_scan(use_label: bool, pred: dict, gt_file):
    ###########################################
    global CLASS_LABELS, eval_class_labels
    if not use_label:
        eval_class_labels = ["class_agnostic"]
    else:
        eval_class_labels = CLASS_LABELS
    ##########################################3
    pred_info = make_pred_info(pred)
    try:
        gt_ids = util_3d.load_ids(gt_file)
        ###tmp add
        sem = (gt_ids//1000)
        ## if reserve all points, but only evaluate chair
        gt_ids[sem!=CHAIR_ID] = -1
        ###
    except Exception as e:
        util.print_error("unable to load " + gt_file + ": " + str(e))
    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    # associate
    if use_label:
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt["matched_pred"] = []
    else:
        gt2pred = {}
        agnostic_instances = []
        # concat all the instances label to agnostic label
        for _, instances in gt_instances.items():
            agnostic_instances += deepcopy(instances)
        for gt in agnostic_instances:
            gt["matched_pred"] = []
        gt2pred[eval_class_labels[0]] = agnostic_instances

    pred2gt = {}
    for label in eval_class_labels:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        if use_label:
            label_id = int(pred_info[uuid]["label_id"])
            if not label_id in ID_TO_LABEL:
                continue
            label_name = ID_TO_LABEL[label_id]
        else:
            label_name = eval_class_labels[0]
        conf = pred_info[uuid]["conf"]
        # read the mask
        pred_mask = pred_info[uuid]["mask"]
        assert len(pred_mask) == len(gt_ids)
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt["min_region_sizes"][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance["uuid"] = uuid
        pred_instance["pred_id"] = num_pred_instances
        pred_instance["label_id"] = label_id if use_label else None
        pred_instance["vert_count"] = num
        pred_instance["confidence"] = conf
        pred_instance["void_intersection"] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask))### True Positive
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)## m每个GT mask会收到一些proposal，有些proposals可能分给多个GT，有些GT没有收到proposals
        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs):
    sep = ""
    col1 = ":"
    lineLen = 100

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>8}".format("AP") + sep
    line += "{:>8}".format("AP_50%") + sep
    line += "{:>8}".format("AP_25%") + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8}".format("RC") + sep
    line += "{:>8}".format("RC_50%") + sep
    line += "{:>8}".format("RC_25%") + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8}".format("PR") + sep
    line += "{:>8}".format("PR_50%") + sep
    line += "{:>8}".format("PR_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(eval_class_labels):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        rc_avg = avgs["classes"][label_name]["rc"]
        rc_50o = avgs["classes"][label_name]["rc50%"]
        rc_25o = avgs["classes"][label_name]["rc25%"]
        pr_avg = avgs["classes"][label_name]["pr"]
        pr_50o = avgs["classes"][label_name]["pr50%"]
        pr_25o = avgs["classes"][label_name]["pr25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>8.3f}".format(ap_avg) + sep
        line += sep + "{:>8.3f}".format(ap_50o) + sep
        line += sep + "{:>8.3f}".format(ap_25o) + sep

        line += "{:>2}".format("|") + sep
        line += sep + "{:>8.3f}".format(rc_avg) + sep
        line += sep + "{:>8.3f}".format(rc_50o) + sep
        line += sep + "{:>8.3f}".format(rc_25o) + sep

        line += "{:>2}".format("|") + sep
        line += sep + "{:>8.3f}".format(pr_avg) + sep
        line += sep + "{:>8.3f}".format(pr_50o) + sep
        line += sep + "{:>8.3f}".format(pr_25o) + sep
        print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]
    all_rc_avg = avgs["all_rc"]
    all_rc_50o = avgs["all_rc_50%"]
    all_rc_25o = avgs["all_rc_25%"]
    all_pr_avg = avgs["all_pr"]
    all_pr_50o = avgs["all_pr_50%"]
    all_pr_25o = avgs["all_pr_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>8.3f}".format(all_ap_avg) + sep
    line += "{:>8.3f}".format(all_ap_50o) + sep
    line += "{:>8.3f}".format(all_ap_25o) + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8.3f}".format(all_rc_avg) + sep
    line += "{:>8.3f}".format(all_rc_50o) + sep
    line += "{:>8.3f}".format(all_rc_25o) + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8.3f}".format(all_pr_avg) + sep
    line += "{:>8.3f}".format(all_pr_50o) + sep
    line += "{:>8.3f}".format(all_pr_25o) + sep
    print(line)
    print("")

def log_results(logger, avgs):
    sep = ""
    col1 = ":"
    lineLen = 100

    logger.info("")
    logger.info("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>8}".format("AP") + sep
    line += "{:>8}".format("AP_50%") + sep
    line += "{:>8}".format("AP_25%") + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8}".format("RC") + sep
    line += "{:>8}".format("RC_50%") + sep
    line += "{:>8}".format("RC_25%") + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8}".format("PR") + sep
    line += "{:>8}".format("PR_50%") + sep
    line += "{:>8}".format("PR_25%") + sep

    logger.info(line)
    logger.info("#" * lineLen)

    for li, label_name in enumerate(eval_class_labels):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        rc_avg = avgs["classes"][label_name]["rc"]
        rc_50o = avgs["classes"][label_name]["rc50%"]
        rc_25o = avgs["classes"][label_name]["rc25%"]
        pr_avg = avgs["classes"][label_name]["pr"]
        pr_50o = avgs["classes"][label_name]["pr50%"]
        pr_25o = avgs["classes"][label_name]["pr25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>8.3f}".format(ap_avg) + sep
        line += sep + "{:>8.3f}".format(ap_50o) + sep
        line += sep + "{:>8.3f}".format(ap_25o) + sep

        line += "{:>2}".format("|") + sep
        line += sep + "{:>8.3f}".format(rc_avg) + sep
        line += sep + "{:>8.3f}".format(rc_50o) + sep
        line += sep + "{:>8.3f}".format(rc_25o) + sep

        line += "{:>2}".format("|") + sep
        line += sep + "{:>8.3f}".format(pr_avg) + sep
        line += sep + "{:>8.3f}".format(pr_50o) + sep
        line += sep + "{:>8.3f}".format(pr_25o) + sep
        logger.info(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]
    all_rc_avg = avgs["all_rc"]
    all_rc_50o = avgs["all_rc_50%"]
    all_rc_25o = avgs["all_rc_25%"]
    all_pr_avg = avgs["all_pr"]
    all_pr_50o = avgs["all_pr_50%"]
    all_pr_25o = avgs["all_pr_25%"]

    logger.info("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>8.3f}".format(all_ap_avg) + sep
    line += "{:>8.3f}".format(all_ap_50o) + sep
    line += "{:>8.3f}".format(all_ap_25o) + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8.3f}".format(all_rc_avg) + sep
    line += "{:>8.3f}".format(all_rc_50o) + sep
    line += "{:>8.3f}".format(all_rc_25o) + sep

    line += "{:>2}".format("|") + sep
    line += "{:>8.3f}".format(all_pr_avg) + sep
    line += "{:>8.3f}".format(all_pr_50o) + sep
    line += "{:>8.3f}".format(all_pr_25o) + sep
    logger.info(line)
    logger.info("#" * lineLen)
    logger.info("")


def write_result_file(avgs, filename):
    _SPLITTER = ","
    with open(filename, "w") as f:
        f.write(_SPLITTER.join(["class", "class id", "ap", "ap50", "ap25"]) + "\n")
        for class_name in eval_class_labels:
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, ap, ap50, ap25]]) + "\n")


def evaluate(use_label: bool, preds: dict, gt: dict, logger=None, log=False, prcurv_save_dir=None):
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global opt

    print("evaluating", len(preds), "scans...")
    matches = {}
    for i, (k, v) in enumerate(preds.items()):
        # gt_file = os.path.join(gt_path, k + ".txt")
        # if not os.path.isfile(gt_file):
        #     util.print_error("Scan {} does not match any gt file".format(k), user_fault=True)

        # matches_key = os.path.abspath(gt_file)
        matches_key = k
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(use_label, v, gt[k])
        matches[matches_key] = {}
        matches[matches_key]["gt"] = gt2pred
        matches[matches_key]["pred"] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print("")
    ap_scores, recall_scores, precision_scores = evaluate_matches(use_label, matches, prcurv_save_dir)
    avgs = compute_averages(use_label, ap_scores, recall_scores, precision_scores)

    # return avgs
    # print
    print_results(avgs)
    if (logger is not None) and log:
        log_results(logger, avgs)
    # write_result_file(avgs, output_file)
