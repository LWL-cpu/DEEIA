import sys
sys.path.append("../")
import copy
import spacy

from utils import _normalize_answer, find_head, hungarian_matcher
from utils import WhitespaceTokenizer
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def eval_rpf(gt_num, pred_num, correct_num):
    recall = correct_num / gt_num if gt_num != 0 else .0
    precision = correct_num / pred_num if pred_num != 0 else .0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 1e-4 else .0
    res = {
        "recall": recall, "precision": precision, "f1": f1,
        "gt_num": gt_num, "pred_num": pred_num, "correct_num": correct_num,
    }
    return res


def eval_std_f1_score(features, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0

    for feature in features:
        for (roles, gt_dict_word, pred_dict_word) in zip(feature.arg_list, feature.gt_dict_word,
                                                         feature.pred_dict_word):
            all_pred_list = list()
            all_gt_list = list()
            for role in roles:
                gt_list = gt_dict_word[role] if role in gt_dict_word else list()
                pred_list = list(set(pred_dict_word[role])) if role in pred_dict_word else list()
                gt_num += len(gt_list)
                pred_num += len(pred_list)

                for gt_span in gt_list:
                    if gt_span in pred_list:
                        correct_num += 1

                all_pred_list.extend(copy.deepcopy(pred_list))
                all_gt_list.extend(gt_list)

            all_pred_list = list(set(all_pred_list))
            all_gt_list = list(set(all_gt_list))
            pred_num_identify += len(all_pred_list)
            gt_num_identify += len(all_gt_list)
            for gt_span in all_gt_list:
                if gt_span in all_pred_list:
                    correct_identify_num += 1

    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_text_f1_score(features, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0

    for feature in features:
        full_text = feature.full_text
        for (roles, gt_dict_word, pred_dict_word) in zip(feature.arg_list, feature.gt_dict_word,
                                                         feature.pred_dict_word):
            all_pred_list = list()
            all_gt_list = list()
            for role in roles:
                gt_list = gt_dict_word[role] if role in gt_dict_word else list()
                pred_list = list(set(pred_dict_word[role])) if role in pred_dict_word else list()
                #################### The only difference with eval_std_f1_score ###############################
                gt_texts = [_normalize_answer(" ".join(full_text[gt_span[0]: gt_span[1] + 1])) for gt_span in gt_list]
                pred_texts = list(
                    set([_normalize_answer(" ".join(full_text[pred_span[0]: pred_span[1] + 1])) for pred_span in
                         copy.deepcopy(pred_list)]))
                gt_list = gt_texts
                pred_list = pred_texts
                #########################################################################################################################################
                gt_num += len(gt_list)
                pred_num += len(pred_list)

                for gt_span in gt_list:
                    if gt_span in pred_list:
                        correct_num += 1

                all_pred_list.extend(copy.deepcopy(pred_list))
                all_gt_list.extend(gt_list)

            all_pred_list = list(set(all_pred_list))
            all_gt_list = list(set(all_gt_list))
            pred_num_identify += len(all_pred_list)
            gt_num_identify += len(all_gt_list)
            for gt_span in all_gt_list:
                if gt_span in all_pred_list:
                    correct_identify_num += 1

        res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
        res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_head_f1_score(features, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0
    last_full_text = None

    for feature in features:
        full_text = feature.full_text
        for (roles, gt_dict_word, pred_dict_word) in zip(feature.arg_list, feature.gt_dict_word,
                                                         feature.pred_dict_word):
            all_pred_list = list()
            all_gt_list = list()
            for role in roles:
                gt_list = gt_dict_word[role] if role in gt_dict_word else list()
                pred_list = list(set(pred_dict_word[role])) if role in pred_dict_word else list()
                #################### The only difference with eval_std_f1_score ###############################
                if full_text != last_full_text:
                    # Reduce the time of doc generation, which is highly time-consuming
                    doc = nlp(" ".join(full_text))
                    last_full_text = full_text

                gt_head_texts = [str(find_head(gt_span[0], gt_span[1] + 1, doc)) for gt_span in gt_list]
                pred_head_texts = list(set([str(find_head(pred_span[0], pred_span[1] + 1, doc)) for pred_span in
                                            copy.deepcopy(pred_list)]))
                gt_list = gt_head_texts
                pred_list = pred_head_texts
                #########################################################################################################################################
                gt_num += len(gt_list)
                pred_num += len(pred_list)

                for gt_span in gt_list:
                    if gt_span in pred_list:
                        correct_num += 1

                all_pred_list.extend(copy.deepcopy(pred_list))
                all_gt_list.extend(gt_list)

            all_pred_list = list(set(all_pred_list))
            all_gt_list = list(set(all_gt_list))
            pred_num_identify += len(all_pred_list)
            gt_num_identify += len(all_gt_list)
            for gt_span in all_gt_list:
                if gt_span in all_pred_list:
                    correct_identify_num += 1

    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def show_results(features, output_file, metainfo):
    """ DEEIA std show resuults """
    with open(output_file, 'w', encoding='utf-8') as f:
        "0: wrong span; 1:partial; 2:overlapping; 3:missing; 4:over extracting"
        error_types = ["wrong span", "partial", "overlapping", "missing", "over extracting"]
        errors = [[] for i in range(5)]

        for k, v in metainfo.items():
            f.write(f"{k}: {v}\n")

        for feature in features:
            example_id = feature.example_id

            sent = feature.enc_text
            full_text = feature.full_text
            f.write("\n-------------------------------------------------------------------------------------\n")
            f.write("Example ID {}\n".format(example_id))
            f.write("Sent: {}\n".format(sent))
            for (event_type, event_trigger, roles, pred_dict_word, gt_dict_word) in \
                    zip(feature.event_type, feature.event_trigger, feature.arg_list, feature.pred_dict_word,
                        feature.gt_dict_word):
                f.write("-------------------------------------------------------------------------------------\n")
                f.write("Event type: {}\t\t\tTrigger word: {}\n".format(event_type, event_trigger))

                for arg_role in roles:

                    pred_list = pred_dict_word[arg_role] if arg_role in pred_dict_word else list()
                    gt_list = gt_dict_word[arg_role] if arg_role in gt_dict_word else list()
                    if len(pred_list) == 0 and len(gt_list) == 0:
                        continue

                    if len(gt_list) == 0 and len(pred_list) > 0:
                        gt_list = [(-1, -1)] * len(pred_list)

                    if len(gt_list) > 0 and len(pred_list) == 0:
                        pred_list = [(-1, -1)] * len(gt_list)

                    gt_idxs, pred_idxs = hungarian_matcher(gt_list, pred_list)

                    for pred_idx, gt_idx in zip(pred_idxs, gt_idxs):
                        if gt_list[gt_idx] == (-1, -1) and pred_list[pred_idx] == (-1, -1):
                            continue
                        else:
                            pred_text = " ".join(full_text[pred_list[pred_idx][0]: pred_list[pred_idx][1] + 1]) if \
                                pred_list[pred_idx] != (-1, -1) else "__ No answer __"
                            gt_text = " ".join(full_text[gt_list[gt_idx][0]: gt_list[gt_idx][1] + 1]) if \
                                gt_list[gt_idx] != (-1,-1) else "__ No answer __"

                        if gt_list[gt_idx] == pred_list[pred_idx]:
                            f.write("Arg {} matched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text,
                                                                                                pred_list[pred_idx][0],
                                                                                                pred_list[pred_idx][1],
                                                                                                gt_text,
                                                                                                gt_list[gt_idx][0],
                                                                                                gt_list[gt_idx][1]))
                        else:
                            if pred_list[pred_idx] == (-1, -1):
                                error_type = 3
                            elif gt_list[gt_idx] == (-1, -1):
                                error_type = 4
                            elif pred_list[pred_idx][0] > gt_list[gt_idx][1] or pred_list[pred_idx][1] < gt_list[gt_idx][0]:
                                error_type = 0
                            elif (pred_list[pred_idx][0] >= gt_list[gt_idx][0] and pred_list[pred_idx][1] <= gt_list[gt_idx][1]) or \
                                    (pred_list[pred_idx][0] <= gt_list[gt_idx][0] and pred_list[pred_idx][1] >= gt_list[gt_idx][1]):
                                error_type = 1
                            else:
                                error_type = 2
                            f.write("Arg {} dismatched: Error type: {}\tPred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role,
                                                                                                                   error_types[error_type],
                                                                                                                   pred_text,
                                                                                                                   pred_list[pred_idx][0],
                                                                                                                   pred_list[pred_idx][1],
                                                                                                                   gt_text,
                                                                                                                   gt_list[gt_idx][0],
                                                                                                                   gt_list[gt_idx][1]))
                            errors[error_type].append([event_type,event_trigger, arg_role, pred_list[pred_idx],
                                                       pred_text, gt_list[gt_idx], gt_text, example_id, sent, full_text])



                    if len(gt_idxs) < len(gt_list):  # prediction  __no answer__
                        for idx in range(len(gt_list)):
                            if idx not in gt_idxs:
                                error_type = 3
                                gt_text = " ".join(full_text[gt_list[idx][0]: gt_list[idx][1] + 1])
                                f.write("Arg {} dismatched: Error type: {}\tPred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role,
                                                                                                       error_types[error_type],
                                                                                                       "__ No answer __",
                                                                                                       -1, -1, gt_text,
                                                                                                       gt_list[idx][0],
                                                                                                       gt_list[idx][1]))
                                errors[error_type].append([event_type, event_trigger, arg_role, pred_list[pred_idx],
                                                           pred_text, gt_list[gt_idx], gt_text, example_id, sent,
                                                           full_text])

                    if len(pred_idxs) < len(pred_list):  # ground truth  __no answer__
                        for idx in range(len(pred_list)):
                            if idx not in pred_idxs:
                                error_type = 4
                                pred_text = " ".join(full_text[pred_list[idx][0]: pred_list[idx][1] + 1])
                                f.write(
                                    "Arg {} dismatched: Error type: {}\tPred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text,
                                                                                                   error_types[error_type],
                                                                                                   pred_list[idx][0],
                                                                                                   pred_list[idx][1],
                                                                                                   "__ No answer __",
                                                                                                   -1, -1))
                                errors[error_type].append([event_type, event_trigger, arg_role, pred_list[pred_idx],
                                                           pred_text, gt_list[gt_idx], gt_text, example_id, sent,
                                                           full_text])

                f.write("-------------------------------------------------------------------------------------\n")

        f.write("\n\n")
        f.write("-------------------------------------------------------------------------------------\n")
        f.write("-------------------------------------------------------------------------------------\n")
        f.write("-------------------------------------------------------------------------------------\n")

        f.write("Error analysis:\n")
        total_errors = sum([len(errors[i]) for i in range(len(errors))])
        f.write(f"Total errors: {total_errors}\n")
        for i in range(len(errors)):
            f.write(f"#{i}.{error_types[i]}: total: {len(errors[i])}, proportion: {len(errors[i]) / total_errors * 100:.2f}%\n")
        for i in range(len(errors)):
            f.write(f"\n################# {i}.{error_types[i]}\n")
            for v in errors[i]:
                f.write(f"error: {error_types[i]}, event type: {v[0]}, event trigger: {v[1]}")
                f.write(f"arg_role: {v[2]}, example id: {v[7]}\n")
                f.write(f"sent: {v[8]}\n")
                f.write(f"pred_list: {v[3]}, pred_text: {v[4]}\n")
                f.write(f"gt_list: {v[5]}, gt_text: {v[6]}\n")
                f.write("\n")

        f.write("-------------------------------------------------------------------------------------\n")
        f.write("-------------------------------------------------------------------------------------\n")
        f.write("-------------------------------------------------------------------------------------\n")
