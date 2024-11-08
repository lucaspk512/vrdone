import os
import json
from collections import defaultdict
import numpy as np

from .prepare_eval_labels import prepare_gts_for_vidor, prepare_gts_for_vidvrd
from VidVRD_helper.evaluation.visual_relation_detection import eval_detection_scores, eval_tagging_scores
from VidVRD_helper.evaluation.common import voc_ap
from dataloaders.category import  vidor_category_id_to_name, vidor_pred_id_to_name
from dataloaders.category import vidvrd_category_id_to_name, vidvrd_pred_id_to_name

class EvaluationFormatConvertor(object):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type.lower()
        if self.dataset_type == "vidvrd":
            self.entity_id_to_name = vidvrd_category_id_to_name
            self.pred_id_to_name = vidvrd_pred_id_to_name
            
        elif self.dataset_type == "vidor":
            self.entity_id_to_name = vidor_category_id_to_name
            self.pred_id_to_name = vidor_pred_id_to_name
        else:
            raise NotImplementedError()

    def _reset_video_name(self, video_name):
        if self.dataset_type == "vidor":
            temp = video_name.split('_')  # e.g., "0001_3598080384"
            assert len(temp) == 2
            video_name = temp[1]
        elif self.dataset_type == "vidvrd":
            # e.g., video_name == "ILSVRC2015_train_00005015"
            pass
        else:
            assert False

        return video_name

    def to_eval_format_pr(self, video_name, pr_triplet):
        '''
        this func is compatible for predictions both before and after grounding
        use_pku: Liu et al, (Peking University), paper: "Beyond Short-Term Snippet: Video Relation Detection with Spatio-Temporal Global Context"
        they have a different id2category map
        '''

        video_name = self._reset_video_name(video_name)
        if len(pr_triplet) is None:
            return {video_name: []}
        
        results_per_video = []
        for p_id in range(len(pr_triplet['triplets'])):
            result_per_triplet = dict()

            s_cat_id, pred_cat_id, o_cat_id = tuple(pr_triplet['triplets'][p_id])

            result_per_triplet["triplet"] = [
                self.entity_id_to_name[s_cat_id],
                self.pred_id_to_name[pred_cat_id],
                self.entity_id_to_name[o_cat_id],
            ]

            dura_ = (pr_triplet["pred_durations"][p_id][0], pr_triplet["pred_durations"][p_id][1])
            result_per_triplet["duration"] = dura_ 
            
            result_per_triplet["score"] = float(pr_triplet["triple_scores_avg"][p_id])

            result_per_triplet["sub_traj"] = pr_triplet["so_trajs"][p_id][0]  # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = pr_triplet["so_trajs"][p_id][1]

            assert (len(pr_triplet["so_trajs"][p_id][0]) == len(pr_triplet["so_trajs"][p_id][1]) and
                    len(pr_triplet["so_trajs"][p_id][1]) == (dura_[1] - dura_[0]))

            results_per_video.append(result_per_triplet)
        return {video_name: results_per_video}    



def eval_visual_relation(groundtruth, prediction, viou_threshold=0.5,
        det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores = eval_detection_scores(
                gt_relations, predict_relations, viou_threshold)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    return mean_ap, rec_at_n, mprec_at_n

def eval_relation(
    dataset_type,
    prediction_results=None,
    json_results_path=None,
    config=None,
):
    if prediction_results is None:
        assert json_results_path is not None
        print("loading json results from {}".format(json_results_path))
        with open(json_results_path,'r') as f:
            prediction_results = json.load(f)
        print("Done.")
    else:
        assert json_results_path is None

    assert config is not None
    gt_relations_path = config['prepare_gt_config']['gt_relations_path']
    assert gt_relations_path is None or gt_relations_path.endswith(".json")
    
    if not os.path.exists(gt_relations_path):
        print("Gt relation path does not exist, then generate.")
        dataset_type = dataset_type.lower()
        assert dataset_type in ["vidvrd", "vidor"]
        if dataset_type == "vidvrd":
            prepare_gts_for_vidvrd(config)
        else:
            prepare_gts_for_vidor(config)
    else:
        print("Gt relation path does exist, then load.")

    with open(gt_relations_path,'r') as f:
        gt_relations = json.load(f)

    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations, prediction_results, viou_threshold=config['inference_config']['viou_th'])
    
    result = {"RelDet_mAP": mean_ap}
    result.update({"RelDet_AR@" + str(k): v for k, v in rec_at_n.items()})
    result.update({"RelTag_AP@" + str(k): v for k, v in mprec_at_n.items()})

    return result