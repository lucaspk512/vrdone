import os
import json
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict
from utils.utils_func import is_overlap, merge_duration_list,linear_interpolation
from utils.categories_v2 import vidvrd_CatName2Id,vidvrd_PredName2Id

from dataloaders.dataloader_vidvrd import TrajProposal, VideoGraph

class VidVRD(object):
    def __init__(
        self,
        split,
        ann_dir,
        proposal_dir,
        dim_boxfeature,
        min_frames_th,
        max_proposal,
        max_preds,
        save_dir,
    ):
        self.split = split
        self.proposal_dir = proposal_dir  # e.g., "proposals/vidvrd-dataset/miss30_minscore0p3/VidVRD_test_every1frames"
        self.dim_boxfeature = dim_boxfeature
        self.min_frames_th = min_frames_th
        self.max_proposal = max_proposal
        self.max_preds = max_preds
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.split == "train":
            self.video_ann_dir = os.path.join(
                ann_dir, 'train'
            )  # e.g., "datasets/vidvrd-dataset/train/"
        else:  # self.split == "test":
            self.video_ann_dir = os.path.join(
                ann_dir, 'test'
            )  # e.g., "datasets/vidvrd-dataset/test/"

        self.video_name_list = self._prepare_video_names()

        print("preparing data...")
        for video_name in tqdm(self.video_name_list):
            if os.path.exists(os.path.join(self.save_dir, video_name+'.pkl')):
                continue

            video_data = self.get_data(video_name)
            # save 

            with open(os.path.join(self.save_dir, video_name+'.pkl'), 'wb') as f:
                pickle.dump(video_data, f)
        print("all data have been saved.")


    def _prepare_video_names(self):
        video_name_list = os.listdir(self.video_ann_dir)
        video_name_list = sorted([v.split('.')[0] for v in video_name_list])
        return video_name_list

    def get_data(self, video_name):
        traj_proposal = self._get_proposal(video_name)
        gt_graph = self._get_gt_graph(video_name)

        traj_proposal.video_len = (
            gt_graph.video_len
        )  # TODO add traj_proposal.video_len separately and assert `traj_proposal.video_len == gt_graph.video_len`
        traj_proposal.video_wh = gt_graph.video_wh
        
        data_dict = dict()
        data_dict['traj_proposal'] = {k:v for k, v in traj_proposal.__dict__.items()}
        data_dict['gt_graph'] = {k:v for k, v in gt_graph.__dict__.items()}

        return data_dict

    def _get_proposal(self, video_name):
        track_res_path = os.path.join(
            self.proposal_dir, video_name + ".npy"
        )  # ILSVRC2015_train_00010001.npy
        track_res = np.load(track_res_path, allow_pickle=True)
        trajs = {box_info[1]: {} for box_info in track_res}
        for tid in trajs.keys():
            trajs[tid]["frame_ids"] = []
            trajs[tid]["bboxes"] = []
            trajs[tid]["roi_features"] = []
            trajs[tid][
                "category_id"
            ] = (
                []
            )  # 如果某个tid只有len==6的box_info，那就无法获取 category_id ，默认为背景

        for box_info in track_res:
            if not isinstance(box_info, list):
                box_info = box_info.tolist()
            assert (
                len(box_info) == 6 or len(box_info) == 12 + self.dim_boxfeature
            ), "len(box_info)=={}".format(len(box_info))

            frame_id = int(box_info[0])
            tid = box_info[1]
            tracklet_xywh = box_info[2:6]
            xmin_t, ymin_t, w_t, h_t = tracklet_xywh
            xmax_t = xmin_t + w_t
            ymax_t = ymin_t + h_t
            bbox_t = [xmin_t, ymin_t, xmax_t, ymax_t]
            confidence = float(0)
            if len(box_info) == 12 + self.dim_boxfeature:
                confidence = box_info[6]
                cat_id = box_info[7]
                xywh = box_info[8:12]
                xmin, ymin, w, h = xywh
                xmax = xmin + w
                ymax = ymin + h
                bbox = [
                    (xmin + xmin_t) / 2,
                    (ymin + ymin_t) / 2,
                    (xmax + xmax_t) / 2,
                    (ymax + ymax_t) / 2,
                ]
                roi_feature = box_info[12:]
                trajs[tid]["category_id"].append(cat_id)
                trajs[tid]["roi_features"].append(roi_feature)

            if len(box_info) == 6:
                bbox_t.append(confidence)
                trajs[tid]["bboxes"].append(bbox_t)
                trajs[tid]["roi_features"].append([0] * self.dim_boxfeature)
            else:
                bbox.append(confidence)
                trajs[tid]["bboxes"].append(bbox)
            trajs[tid]["frame_ids"].append(frame_id)  #

        for tid in trajs.keys():
            if trajs[tid]["category_id"] == []:
                trajs[tid]["category_id"] = 0
            else:
                # print(trajs[tid]["category_id"])
                temp = np.argmax(np.bincount(trajs[tid]["category_id"]))  # 求众数
                trajs[tid]["category_id"] = int(temp)

            frame_ids = trajs[tid]["frame_ids"]
            start = min(frame_ids)
            end = max(frame_ids) + 1
            dura_len = end - start
            duration = (start, end)  # 前闭后开区间
            trajs[tid]["roi_features"] = np.array(trajs[tid]["roi_features"])
            trajs[tid]["bboxes"] = np.array(trajs[tid]["bboxes"])

            # 将太短的视为背景，后续过滤掉
            if len(frame_ids) < self.min_frames_th:
                trajs[tid]["category_id"] = 0
            else:
                trajs[tid]["duration"] = (start, end)

            # 对于非背景的traj， 看是否需要插值
            if trajs[tid]["category_id"] != 0 and len(frame_ids) != dura_len:
                trajs[tid]["roi_features"] = linear_interpolation(
                    trajs[tid]["roi_features"], frame_ids
                )
                trajs[tid]["bboxes"] = linear_interpolation(
                    trajs[tid]["bboxes"], frame_ids
                )

            if trajs[tid]["category_id"] != 0:
                assert len(trajs[tid]["bboxes"]) == dura_len

        # trajs = {k:v for k,v in trajs.items() if v["category_id"]!=0}
        cat_ids = []
        traj_boxes = []
        roi_features_list = []
        traj_durations = []
        for tid in trajs.keys():
            if trajs[tid]["category_id"] != 0:
                dura_len = trajs[tid]["duration"][1] - trajs[tid]["duration"][0]
                assert len(trajs[tid]["bboxes"]) == dura_len
                cat_ids.append(trajs[tid]["category_id"])
                traj_boxes.append(trajs[tid]["bboxes"])
                roi_features_list.append(trajs[tid]["roi_features"])
                traj_durations.append(trajs[tid]["duration"])

        return TrajProposal(
            video_name,
            cat_ids,
            traj_boxes,
            traj_durations,
            roi_features_list,
            self.max_proposal,
        )

    def _get_gt_graph(self, video_name):
        video_ann_path = os.path.join(self.video_ann_dir, video_name + ".json")

        ## 1. construct trajectory annotations from frame-level bbox annos
        if os.path.exists(video_ann_path):
            with open(video_ann_path, 'r') as f:
                video_anno = json.load(f)
        else:
            print(video_name, "not find its anno")
            raise NotImplementedError

        video_len = len(video_anno["trajectories"])
        video_wh = (video_anno["width"], video_anno["height"])

        traj_categories = video_anno[
            "subject/objects"
        ]  # tid not necessary 0 ~ len(traj_categories)-1
        # tid2category_map = [traj["category"] for traj in traj_categories] #  This is WRONG!
        tid2category_map = {
            traj["tid"]: traj["category"] for traj in traj_categories
        }  # this is CORRECT
        # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}
        trajs = {traj["tid"]: {} for traj in traj_categories}

        for tid in trajs.keys():
            trajs[tid]["all_bboxes"] = []
            trajs[tid]["frame_ids"] = []
            trajs[tid]["cat_name"] = tid2category_map[tid]

        for frame_id, frame_anno in enumerate(video_anno["trajectories"]):
            for bbox_anno in frame_anno:
                tid = bbox_anno["tid"]
                category_name = tid2category_map[tid]
                category_id = vidvrd_CatName2Id[category_name]

                bbox = bbox_anno["bbox"]
                bbox = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
                trajs[tid]["all_bboxes"].append(bbox)
                trajs[tid]["frame_ids"].append(frame_id)
                trajs[tid]["category_id"] = category_id

        ## 2.linear_interpolation
        for tid in trajs.keys():
            frame_ids = trajs[tid]["frame_ids"]
            start, end = min(frame_ids), max(frame_ids) + 1  # 前开后闭区间
            all_bboxes = np.array(trajs[tid]["all_bboxes"]).astype(np.float32)
            all_bboxes = linear_interpolation(all_bboxes, frame_ids)
            trajs[tid]["all_bboxes"] = all_bboxes  # np.ndarray, shape == (dura_len,4)
            trajs[tid]["duration"] = (start, end)  # dura_len == end - start

        traj_cat_ids = []
        traj_durations = []
        traj_bbox_list = []
        tid2idx_map = {}
        for idx, tid in enumerate(trajs.keys()):
            traj_cat_ids.append(trajs[tid]["category_id"])
            traj_durations.append(trajs[tid]["duration"])
            traj_bbox_list.append(trajs[tid]["all_bboxes"])
            tid2idx_map[tid] = idx
        traj_cat_ids = np.array(traj_cat_ids)  # shape == (num_trajs,)
        traj_durations = np.array(traj_durations)  # shape == (num_trajs,2)
        num_trajs = len(traj_cat_ids)

        # 3. merge relations
        # in the train-set of vidvrd, some long relations is annotated as short segments
        # we merge them to one whole relation
        # e.g., a original relation might have a duration_list = [(195, 225), (225, 240), (375, 405), (390, 420)]
        # we merge it to [(195, 240), (375, 420)]
        # NOTE: train 的需要merge， 在vidvrd的train-set中， 一个大于30的duration都没有，在test-set中， long-duration没有被按照30一段标注
        preds = video_anno["relation_instances"]
        pred_cat_ids = []
        pred_durations = []
        trituple_list = []
        trituple2durations_dict = defaultdict(list)
        for pred in preds:
            predicate = pred["predicate"]
            subject_tid = pred["subject_tid"]
            object_tid = pred["object_tid"]
            trituple = str(subject_tid) + "-" + predicate + "-" + str(object_tid)

            begin_fid = pred["begin_fid"]
            end_fid = pred["end_fid"]
            trituple2durations_dict[trituple].append((begin_fid, end_fid))

        for trituple, durations in trituple2durations_dict.items():
            merged_durations = merge_duration_list(
                durations
            )  # e.g., [(30,60),(60,90),(120,150)] --> [(30,90),(120,150)]
            trituple2durations_dict[trituple] = merged_durations

            pred_name = trituple.split('-')[1]
            pred_catid = vidvrd_PredName2Id[pred_name]

            for duration in merged_durations:
                trituple_list.append(trituple)
                pred_cat_ids.append(pred_catid)
                pred_durations.append(duration)

        num_preds = len(pred_cat_ids)
        pred_cat_ids = np.array(pred_cat_ids)  # shape == (num_preds,)
        pred_durations = np.array(pred_durations)  # shape == (num_preds,2)

        # 2.3. construct adjacency matrix
        adj_matrix_subject = np.zeros((num_preds, num_trajs), dtype=np.int64)
        adj_matrix_object = np.zeros((num_preds, num_trajs), dtype=np.int64)
        for idx in range(num_preds):
            trituple = trituple_list[idx]
            pred_duration = pred_durations[idx]

            subj_tid = int(trituple.split('-')[0])
            obj_tid = int(trituple.split('-')[-1])
            subj_idx = tid2idx_map[subj_tid]
            obj_idx = tid2idx_map[obj_tid]

            subj_duration = traj_durations[subj_idx]
            if is_overlap(pred_duration, subj_duration):
                adj_matrix_subject[idx, subj_idx] = 1

            obj_duration = traj_durations[obj_idx]
            if is_overlap(
                pred_duration, obj_duration
            ):  # is_overlap 可以用于 1-d np.ndarray
                adj_matrix_object[idx, obj_idx] = 1

        for row in adj_matrix_subject:
            assert np.sum(row) == 1, "video:{} not correct".format(video_name)

        for row in adj_matrix_object:
            assert np.sum(row) == 1, "video:{} not correct".format(video_name)

        video_info = (video_name, video_len, video_wh)

        return VideoGraph(
            video_info,
            self.split,
            traj_cat_ids,
            traj_durations,
            traj_bbox_list,
            pred_cat_ids,
            pred_durations,
            adj_matrix_subject,
            adj_matrix_object,
            self.max_preds,
        )


if __name__ == '__main__':

    vidvrd_dataset = VidVRD(
        split="test",
        ann_dir="../vidvrd/annotations",
        proposal_dir="../vidvrd/features/VidVRD_test_every1frames",
        dim_boxfeature=1024,
        min_frames_th=5,
        max_proposal=50,
        max_preds=100,
        save_dir="../vidvrd/features/vidvrd_per_video_val",
    )
