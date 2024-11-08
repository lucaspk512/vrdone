import os
import pickle
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from copy import deepcopy

import torch

from datasets.category import vidvrd_CatName2Id, vidvrd_CatId2name
from mega_core.structures.bounding_box import BoxList

class VidVRDDatasetGt(torch.utils.data.Dataset):
    def __init__(
        self, 
        cfg,
        data_name, 
        img_dir, 
        anno_path, 
        transforms, 
        save_dir,
        part_id, 
    ):
        self.is_train = False
        self.cfg = cfg
        self.data_name = data_name     
        self.transforms = transforms
        
        self.img_dir = img_dir  
        self.anno_path = anno_path  
        self.save_dir = save_dir
        
        self.video_names = sorted([fn.strip().split('.json')[0] for fn in os.listdir(self.anno_path)])[part_id*100: (part_id+1)*100]

        self.video_num_frames = [len(os.listdir(os.path.join(self.img_dir, vn))) for vn in self.video_names]

        self._img_dir = os.path.join(self.img_dir, "%s/%s.jpg")  
        self._anno_path = os.path.join(self.anno_path, "%s.json")
        self.image_set_index = []
        for vid, vn in enumerate(self.video_names):
            self.image_set_index += [os.path.join(vn, vn + "_{:06d}".format(frame_idx)) for frame_idx in range(1, self.video_num_frames[vid] + 1)]

        keep = self.filter_annotation()
        
        self.pattern = [vn + '/' + '_'.join([vn, "%06d"]) for vn in self.video_names]
        frame_seg_len = []
        frame_seg_id = []
        pattern = []
        for vidx, nf in enumerate(self.video_num_frames):
            frame_seg_len += [nf] * nf
            frame_seg_id += list(range(1, nf+1))
            pattern += [self.pattern[vidx]] * nf
            
        self.frame_seg_len = frame_seg_len
        self.frame_seg_id = frame_seg_id
        self.pattern = pattern
        assert len(self.image_set_index) == len(self.frame_seg_len) and len(self.frame_seg_len) == len(self.frame_seg_id)
        assert len(self.frame_seg_id) == len(self.pattern)

        self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
        self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self._prepare_video_frame_infos()
        self.Name2Id = vidvrd_CatName2Id

        self.annos = self.load_annos()  
        
        assert len(self.annos) == len(self.image_set_index)
        assert len(set(self.image_set_index)) == len(self.image_set_index)
        
        if not self.is_train:  # from vid_mega.py
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                temp = image_index.split("/")
                video_name, frame_id = temp[0], temp[1].split('_')[-1]
                frame_ids_list = self.filtered_frame_idx[video_name]
                start_frame_id = frame_ids_list[0]

                frame_id = int(frame_id)
                # if frame_id == 0:
                #     如果 frame_id == 0 这一帧被过滤掉了怎么办？ 那这个视频不是被当成上一个视频了吗？（即，这个视频的每一帧记录的 start_id 都是上一个视频的start_id）
                if frame_id == start_frame_id:
                    # 这个 id 是全局的，所以start_id 就是记录每个video是从哪个id开始的
                    self.start_index.append(id)  
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        # shuffled_index = np.arange(self.frame_seg_len[id]) # shuffled_index 里存的是 frame_id
                        # frame_seg_len[id] 的长度，是 filter 之前的长度，我们现在要搞成 filter之后的，把filter掉的不算进去
                        # 在原来的代码中，这样是没问题的，就是说，在原来的代码中，shuffled_index 中的 index 对应的图片，可以没有 annotation，因为 proposal是用 rpn提的
                        # 但是在我们这里，把gt的标注作为 proposal， 所以应该 shuffled_index 中的每一个index对应的图片都要有 annotation
                        # 所以我们的 shuffled_index 应该设置为过滤之后的 frame_id
                        shuffled_index = np.array(frame_ids_list)
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        
                        # 每一个起始帧，都有一个 shuffled_index,
                        self.shuffled_index[str(id)] = shuffled_index
                        # self.shuffled_index[video_name] = shuffled_index # 等价于每个视频都有一个 shuffled_index, 这样写可读性更强 （但是呢，原来的代码等不动就尽量不动，万一有点什么问题）

                    self.start_id.append(id)
                else:
                    assert frame_id > start_frame_id
                    self.start_id.append(self.start_index[-1])
                    

    def __getitem__(self, idx):  # maintain
        return self._get_test(idx)
    
    def __len__(self):  # maintain
        return len(self.image_set_index)
     
    def filter_annotation(self):
        keep = np.zeros((len(self.image_set_index)), dtype=bool)
        outer_idx = 0
        for vid, vn in enumerate(self.video_names):
            with open(self._anno_path % vn, 'r') as json_file:
                video_ann = json.load(json_file)

            frame_count = self.video_num_frames[vid]
            assert frame_count >= int(video_ann['frame_count'])

            for frame_id in range(frame_count):
                if frame_id < len(video_ann['trajectories']) and len(video_ann["trajectories"][frame_id]) > 0:
                    keep[outer_idx] = True
                else:
                    keep[outer_idx] = False

                outer_idx += 1
        
        assert outer_idx == len(self.image_set_index)
        return keep

    def _prepare_video_frame_infos(self):
        video_name_dup_list = [x.split('/')[0] for x in self.image_set_index]  
        self.video_frame_count = {x: video_name_dup_list.count(x) for x in self.video_names}

        self.filtered_frame_idx = {}
        temp = [x.split('/') for x in self.image_set_index] 
        for vn in tqdm(self.video_names):
            frame_ids = sorted([int(x[1].split('_')[-1]) for x in temp if x[0]==vn])
            self.filtered_frame_idx[vn] = frame_ids
        
    def _preprocess_annotation(self, objs, tid2category_map, width_height, video_name, frame_id):
        boxes = []
        gt_classes_tensor = []
        tids_tensor = []
        im_info = width_height  # a tuple of width and height
        for obj in objs:
            bbox = obj["bbox"]
            bbox = [
                np.maximum(float(bbox["xmin"]), 0),
                np.maximum(float(bbox["ymin"]), 0),
                np.minimum(float(bbox["xmax"]), im_info[0] - 1),  # NOTE im_info is w,h
                np.minimum(float(bbox["ymax"]), im_info[1] - 1),
            ]
            boxes.append(bbox)
            gt_class = tid2category_map[obj["tid"]]
            gt_classes_tensor.append(self.Name2Id[gt_class])
            tids_tensor.append(obj["tid"])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # shape == (num_box,)
        tids_tensor = torch.tensor(tids_tensor)  # shape == (num_box,)
        gt_classes_tensor = torch.tensor(gt_classes_tensor)
        # print(boxes_tensor.to(torch.int64),"in dataset")
        res = {
            "boxes": boxes_tensor,  # MARK
            "labels": gt_classes_tensor,
            "tids": tids_tensor,
            "im_info": im_info,
            "video_name": video_name,
            "frame_id": frame_id
        }
        
        return res
        
    def load_annos(self):
        annos = []
        self.raw_annos = {}
        for vn in self.video_names:
            with open(self._anno_path % vn, 'r') as json_file:
                video_ann = json.load(json_file)
            
            self.raw_annos[vn] = video_ann
            width_height = (int(video_ann["width"]), int(video_ann["height"]))
            frame_ids = [fid - 1 for fid in self.filtered_frame_idx[vn]]

            traj_categories = video_ann["subject/objects"]  
            # tid2category_map = [traj["category"] for traj in traj_categories] #  这样写是不对的, tid 未必从 0 ~ len(traj_categories)-1 都有
            tid2category_map = {traj["tid"]: traj["category"] for traj in traj_categories}  # 要这样搞
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}
            for frame_id in frame_ids:
                objs = video_ann["trajectories"][frame_id]  # 长度为当前frame中bbox的数量
                anno = self._preprocess_annotation(objs, tid2category_map, width_height, vn, frame_id)
                annos.append(anno) 
        return annos
    
    def check_anno(self, raw_bbox, video_name, frame_id):
        objs = self.raw_annos[video_name]['trajectories'][frame_id]
        width, height = int(self.raw_annos[video_name]["width"]), int(self.raw_annos[video_name]["height"])

        boxes = []

        for obj in objs:
            bbox = obj["bbox"]
            bbox = [
                np.maximum(float(bbox["xmin"]), 0),
                np.maximum(float(bbox["ymin"]), 0),
                np.minimum(float(bbox["xmax"]), width - 1),  # NOTE im_info is w,h
                np.minimum(float(bbox["ymax"]), height - 1),
            ]

            boxes.append(bbox)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        assert boxes_tensor.shape == raw_bbox.shape
        assert np.sum((boxes_tensor - raw_bbox).numpy()) == 0
    
    def _get_test(self, idx):  
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % tuple(filename.split('/'))).convert("RGB")
        
        temp = filename.split("/")
        video_name, frame_id = temp[0], temp[1].split('_')[-1]
        
        frame_id = int(frame_id)
        frame_ids_list = self.filtered_frame_idx[video_name]
        start_frame_id = frame_ids_list[0]
        frame_category = 0
        if frame_id != start_frame_id:  # start frame
            frame_category = 1
      
      
        img_refs_l = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        # ref_id = min(self.frame_seg_len[idx] - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
        # frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET 这一帧不一定在 frame_ids_list 里面
        ref_relative_id = min(
            len(frame_ids_list) - 1,
            frame_ids_list.index(frame_id) + self.cfg.MODEL.VID.MEGA.MAX_OFFSET, # 12
        )

        ref_id = frame_ids_list[ref_relative_id]
        ref_filename = self.pattern[idx] % (ref_id)
        img_ref = Image.open(self._img_dir % tuple(filename.split('/'))).convert("RGB")
        target_ref = self.get_groundtruth(self.image_set_index.index(ref_filename))
        
        self.check_anno(deepcopy(target_ref.bbox), video_name, ref_id-1)

        img_refs_l.append((img_ref, target_ref))

        img_refs_g = []
        if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:  #
            # GLOBAL.SIZE == 10
            size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == start_frame_id else 1 
            shuffled_index = self.shuffled_index[str(self.start_id[idx])]
            # shuffled_index = self.shuffled_index[video_name]  # 这样写可读性更强， 与上面等价
            for id in range(size):
                temp = (idx - self.start_id[idx] + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1)

                # 这个 temp 是相对位置
                # 既然是从shuffled_index中取出来，为什么还要搞的temp 按照顺序呢？
                # 猜测： 这个 temp 是为了把 frame_id==0 和 frame_id >0 区分开， 同时也是确保每一帧取的 temp值不一样
                # frame_id == 0时： size == GLOBAL.SIZE
                # idx - self.start_id[idx] ==0,  id 从 0~ size -1 ， 那 temp= GLOBAL.SIZE - id -1 就从 GLOBAL.SIZE-1 变化到0
                # frame_id > 0 时， size == 1, 这里的for 循环只执行一次， id=0, GLOBAL.SIZE - id - 1 == GLOBAL.SIZE  - 1
                # idx - self.start_id[idx] 是相对起始帧的偏移量， (注意：idx - self.start_id[idx] 不一定等于 frame_id， 对于有些帧没有annotation的被过滤掉的就不等于)
                # 然后 temp 就从 GLOBAL.SIZE  - 1 变化到  GLOBAL.SIZE  - 1 + total_frames, 其中 total_frames 是过滤之后的总帧数
                # 这样一来，就把起始帧和非起始帧取的 global memory 区分开了，同时也把每一帧的 global_memory区分开了

                # print(temp,len(shuffled_index))
                temp_mod = temp % len(frame_ids_list)  # 取 mod

                filename_g = self.pattern[idx] % shuffled_index[temp_mod]
                img_ref = Image.open(self._img_dir % tuple(filename_g.split('/'))).convert("RGB")
                target_ref = self.get_groundtruth(self.image_set_index.index(filename_g))
                
                self.check_anno(deepcopy(target_ref.bbox), video_name, shuffled_index[temp_mod]-1)

                # Problem: 如果 filename_g 不在 self.image_set_index 中怎么办, i.e, 它被 filter_annotation 过滤掉了， 怎么办？
                # 因为在原来的代码中， 是不用获取 filename_g 的 target的
                # 先看一下 self.annos 里每个 video 被过滤的情况
                img_refs_g.append((img_ref, target_ref))

        target = self.get_groundtruth(idx)  # target is BoxList

        self.check_anno(deepcopy(target.bbox), video_name, frame_id-1)

        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"bef",target)

        target = target.clip_to_image(remove_empty=True)

        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"aft",target)
        if self.transforms is not None:
            # 对于 target的 transform， 是根据对图片的resize来做target的resize
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i] = self.transforms(img_refs_l[i][0], img_refs_l[i][1])
            for i in range(len(img_refs_g)):
                img_refs_g[i] = self.transforms(img_refs_g[i][0], img_refs_g[i][1])
        else:
            assert False, "should use transforms"


        images = {}
        images["cur"] = (img, target)  # img 是tensor ， transforms 的最后有to_tensor 的操作
        images["ref_l"] = img_refs_l  # list[tuple], each one is (img_ref,target_ref)
        images["ref_g"] = img_refs_g  # list[tuple], each one is (img_g,target_g)
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["pattern"] = self.pattern[idx]  # # e.g., 0000_2401075277/%06d
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms
        images["filename"] = filename
        return images, idx

    # safe
    def get_groundtruth(self, idx):
        anno = deepcopy(self.annos[idx])
        width, height = anno["im_info"]  # NOTE im_info is w,h
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("tids", anno["tids"])
        return target

    # safe
    def get_img_info(self, idx):
        im_info = deepcopy(self.annos[idx]["im_info"])
        return {"width": im_info[0], "height": im_info[1]}
  

class VidVRDDatasetProposal(torch.utils.data.Dataset):
    def __init__(
        self,  
        cfg,
        img_dir, 
        proposal_path, 
        anno_path,
        transforms, 
        remove_missed=False,
    ):
        self.is_train = False
        self.cfg = cfg
        self.transforms = transforms
        
        self.img_dir = img_dir  
        self.proposal_path = proposal_path  
        self.anno_path = anno_path
        
        self.video_names = sorted([vn.split('.')[0] for vn in os.listdir(self.anno_path)])
        if remove_missed:
            self.video_names.remove('ILSVRC2015_train_00884000')

        self.video_num_frames = [len(os.listdir(os.path.join(self.img_dir, vn))) for vn in self.video_names]
        
        self.image_set_index = []
        for vid, vn in enumerate(self.video_names):
            self.image_set_index += [os.path.join(vn, vn + "_{:06d}".format(frame_idx)) for frame_idx in range(1, self.video_num_frames[vid] + 1)]
        
        keep = self.filter_annotation()
        self.pattern = [vn + '/' + '_'.join([vn, "%06d"]) for vn in self.video_names]
        
        frame_seg_len = []
        frame_seg_id = []
        pattern = []
        for vidx, nf in enumerate(self.video_num_frames):
            frame_seg_len += [nf] * nf
            frame_seg_id += list(range(1, nf+1))
            pattern += [self.pattern[vidx]] * nf
        
        self.frame_seg_len = frame_seg_len
        self.frame_seg_id = frame_seg_id
        self.pattern = pattern
        assert len(self.image_set_index) == len(self.frame_seg_len) and len(self.frame_seg_len) == len(self.frame_seg_id)
        assert len(self.frame_seg_id) == len(self.pattern)
        
        self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
        self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]
        
        self._prepare_video_frame_infos()
        self.Name2Id = vidvrd_CatName2Id
        self.Id2Name = vidvrd_CatId2name
        
        self.infos = self.load_infos()  
        assert len(self.infos) == len(self.image_set_index)
        assert len(set(self.image_set_index)) == len(self.image_set_index)
        
        self._img_dir = os.path.join(self.img_dir, "%s/%s.jpg")  

        if not self.is_train:  # from vid_mega.py
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                temp = image_index.split("/")
                video_name, frame_id = temp[0], temp[1].split('_')[-1]
                frame_ids_list = self.filtered_frame_idx[video_name]
                start_frame_id = frame_ids_list[0]

                frame_id = int(frame_id)
                # if frame_id == 0:
                #     如果 frame_id == 0 这一帧被过滤掉了怎么办？ 那这个视频不是被当成上一个视频了吗？（即，这个视频的每一帧记录的 start_id 都是上一个视频的start_id）
                if frame_id == start_frame_id:
                    # 这个 id 是全局的，所以start_id 就是记录每个video是从哪个id开始的
                    self.start_index.append(id)  
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        # shuffled_index = np.arange(self.frame_seg_len[id]) # shuffled_index 里存的是 frame_id
                        # frame_seg_len[id] 的长度，是 filter 之前的长度，我们现在要搞成 filter之后的，把filter掉的不算进去
                        # 在原来的代码中，这样是没问题的，就是说，在原来的代码中，shuffled_index 中的 index 对应的图片，可以没有 annotation，因为 proposal是用 rpn提的
                        # 但是在我们这里，把gt的标注作为 proposal， 所以应该 shuffled_index 中的每一个index对应的图片都要有 annotation
                        # 所以我们的 shuffled_index 应该设置为过滤之后的 frame_id
                        shuffled_index = np.array(frame_ids_list)
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        
                        # 每一个起始帧，都有一个 shuffled_index,
                        self.shuffled_index[str(id)] = shuffled_index
                        # self.shuffled_index[video_name] = shuffled_index # 等价于每个视频都有一个 shuffled_index, 这样写可读性更强 （但是呢，原来的代码等不动就尽量不动，万一有点什么问题）

                    self.start_id.append(id)
                else:
                    assert frame_id > start_frame_id
                    self.start_id.append(self.start_index[-1])
                    

    def __getitem__(self, idx):  # maintain
        return self._get_test(idx)
    
    def __len__(self):  # maintain
        return len(self.image_set_index)
     
    def filter_annotation(self):
        keep = np.zeros((len(self.image_set_index)), dtype=bool)
        outer_idx = 0
        for vid, vn in enumerate(self.video_names):
            with open(os.path.join(self.proposal_path, vn + '.pkl'), 'rb') as pkl_file:
                proposal_data_dict = pickle.load(pkl_file)['traj_proposal']
            
            frame_count = self.video_num_frames[vid]
            assert frame_count >= int(proposal_data_dict['video_len'])
            
            traj_durations = proposal_data_dict['traj_durations']
            traj_durations[:, 0] -= 1

            min_frame = torch.min(traj_durations).item()
            max_frame = torch.max(traj_durations).item()

            assert min_frame < max_frame
            assert max_frame <= frame_count
            
            for frame_id in range(frame_count):
                if frame_id >= min_frame and frame_id < max_frame:
                    keep[outer_idx] = True
                else:
                    keep[outer_idx] = False

                outer_idx += 1
        
        assert outer_idx == len(self.image_set_index)
        return keep

    def _prepare_video_frame_infos(self):
        video_name_dup_list = [x.split('/')[0] for x in self.image_set_index]  
        self.video_frame_count = {x: video_name_dup_list.count(x) for x in self.video_names}

        self.filtered_frame_idx = {}
        temp = [x.split('/') for x in self.image_set_index] 
        for vn in tqdm(self.video_names):
            frame_ids = sorted([int(x[1].split('_')[-1]) for x in temp if x[0]==vn])
            self.filtered_frame_idx[vn] = frame_ids
        
    def _preprocess_info(self, objs, tid2category_map, width_height, video_name, frame_id):
        boxes = []
        gt_classes_tensor = []
        tids_tensor = []
        im_info = width_height  # a tuple of width and height
        for obj_idx, obj_bbox in objs.items():
            bbox = [
                np.maximum(obj_bbox[0], 0),
                np.maximum(obj_bbox[1], 0),
                np.minimum(obj_bbox[2], im_info[0]-1),  # NOTE im_info is w,h
                np.minimum(obj_bbox[3], im_info[1]-1),
            ]
            boxes.append(bbox)

            gt_classes_tensor.append(tid2category_map[obj_idx])
            tids_tensor.append(obj_idx)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # shape == (num_box,)
        tids_tensor = torch.tensor(tids_tensor)  # shape == (num_box,)
        gt_classes_tensor = torch.tensor(gt_classes_tensor)
        # print(boxes_tensor.to(torch.int64),"in dataset")
        res = {
            "boxes": boxes_tensor,  # MARK
            "labels": gt_classes_tensor,
            "tids": tids_tensor,
            "im_info": im_info,
            "video_name": video_name,
            "frame_id": frame_id
        }
        return res
        
    def load_infos(self):
        infos = []
        self.raw_infos = {}
        for vid, vn in enumerate(self.video_names):
            with open(os.path.join(self.proposal_path, vn + '.pkl'), 'rb') as pkl_file:
                proposal_data_dict = pickle.load(pkl_file)['traj_proposal']

            self.raw_infos[vn] = {
                "video_name": proposal_data_dict['video_name'],
                "cat_ids": proposal_data_dict['cat_ids'],
                "bboxes_list": proposal_data_dict['bboxes_list'],
                "traj_durations": proposal_data_dict['traj_durations'],
                "num_proposals": proposal_data_dict['num_proposals'],
                "video_len": proposal_data_dict['video_len'],
                "video_wh": proposal_data_dict['video_wh'],
            }
            
            proposal_data_dict_copyed = deepcopy(proposal_data_dict)
            
            width_height = proposal_data_dict_copyed['video_wh']
            frame_ids = [fid - 1 for fid in self.filtered_frame_idx[vn]]
            
            entity_traj_durations = proposal_data_dict_copyed['traj_durations']
            entity_traj_durations[:, 0] -= 1
            entity_traj_durations = entity_traj_durations.tolist()
            
            tid2category_map = {idx: cat_id for idx, cat_id in zip(range(proposal_data_dict_copyed['num_proposals']), proposal_data_dict_copyed['cat_ids'].tolist())}
            
            for frame_id in frame_ids:
                objs = {}
                for entity_idx, dura in enumerate(entity_traj_durations):
                    if frame_id >= dura[0] and frame_id < dura[1]:
                        bbox = proposal_data_dict_copyed['bboxes_list'][entity_idx][frame_id - dura[0]].tolist()
                        objs[entity_idx] = bbox
                
                info = self._preprocess_info(objs, tid2category_map, width_height, vn, frame_id)
                infos.append(info) 
            
        return infos
            
    def check_info(self, raw_bbox, video_name, frame_id):
        info = deepcopy(self.raw_infos)
        entity_traj_durations = info[video_name]['traj_durations']
        entity_traj_durations[:, 0] -= 1
        entity_traj_durations = entity_traj_durations.tolist()
        
        objs = {}
        for entity_idx, dura in enumerate(entity_traj_durations):
            if frame_id >= dura[0] and frame_id < dura[1]:
                bbox = info[video_name]['bboxes_list'][entity_idx][frame_id - dura[0]].tolist()
                objs[entity_idx] = bbox
                    
        width, height = info[video_name]["video_wh"]
        boxes = []
        for obj_idx, obj_bbox in objs.items():
            bbox = [
                np.maximum(obj_bbox[0], 0),
                np.maximum(obj_bbox[1], 0),
                np.minimum(obj_bbox[2], width-1),  # NOTE im_info is w,h
                np.minimum(obj_bbox[3], height-1),
            ]
            boxes.append(bbox)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        assert boxes_tensor.shape == raw_bbox.shape
        assert np.sum((boxes_tensor - raw_bbox).numpy()) == 0
            
    def _get_test(self, idx):  
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % tuple(filename.split('/'))).convert("RGB")
        
        temp = filename.split("/")
        video_name, frame_id = temp[0], temp[1].split('_')[-1]
        
        frame_id = int(frame_id)
        frame_ids_list = self.filtered_frame_idx[video_name]
        start_frame_id = frame_ids_list[0]
        
        frame_category = 0
        if frame_id != start_frame_id:  # start frame
            frame_category = 1
      
        img_refs_l = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        # ref_id = min(self.frame_seg_len[idx] - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
        # frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET 这一帧不一定在 frame_ids_list 里面
        ref_relative_id = min(
            len(frame_ids_list) - 1,
            frame_ids_list.index(frame_id) + self.cfg.MODEL.VID.MEGA.MAX_OFFSET, # 12
        )

        ref_id = frame_ids_list[ref_relative_id]
        ref_filename = self.pattern[idx] % (ref_id)
        img_ref = Image.open(self._img_dir % tuple(filename.split('/'))).convert("RGB")
        target_ref = self.get_groundtruth(self.image_set_index.index(ref_filename))
        
        self.check_info(deepcopy(target_ref.bbox), video_name, ref_id-1)

        img_refs_l.append((img_ref, target_ref))

        img_refs_g = []
        if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:  #
            # GLOBAL.SIZE == 10
            size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == start_frame_id else 1 
            shuffled_index = self.shuffled_index[str(self.start_id[idx])]
            # shuffled_index = self.shuffled_index[video_name]  # 这样写可读性更强， 与上面等价
            for id in range(size):
                temp = (idx - self.start_id[idx] + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1)

                # 这个 temp 是相对位置
                # 既然是从shuffled_index中取出来，为什么还要搞的temp 按照顺序呢？
                # 猜测： 这个 temp 是为了把 frame_id==0 和 frame_id >0 区分开， 同时也是确保每一帧取的 temp值不一样
                # frame_id == 0时： size == GLOBAL.SIZE
                # idx - self.start_id[idx] ==0,  id 从 0~ size -1 ， 那 temp= GLOBAL.SIZE - id -1 就从 GLOBAL.SIZE-1 变化到0
                # frame_id > 0 时， size == 1, 这里的for 循环只执行一次， id=0, GLOBAL.SIZE - id - 1 == GLOBAL.SIZE  - 1
                # idx - self.start_id[idx] 是相对起始帧的偏移量， (注意：idx - self.start_id[idx] 不一定等于 frame_id， 对于有些帧没有annotation的被过滤掉的就不等于)
                # 然后 temp 就从 GLOBAL.SIZE  - 1 变化到  GLOBAL.SIZE  - 1 + total_frames, 其中 total_frames 是过滤之后的总帧数
                # 这样一来，就把起始帧和非起始帧取的 global memory 区分开了，同时也把每一帧的 global_memory区分开了

                # print(temp,len(shuffled_index))
                temp_mod = temp % len(frame_ids_list)  # 取 mod

                filename_g = self.pattern[idx] % shuffled_index[temp_mod]
                img_ref = Image.open(self._img_dir % tuple(filename_g.split('/'))).convert("RGB")
                target_ref = self.get_groundtruth(self.image_set_index.index(filename_g))
                
                self.check_info(deepcopy(target_ref.bbox), video_name, shuffled_index[temp_mod]-1)

                # Problem: 如果 filename_g 不在 self.image_set_index 中怎么办, i.e, 它被 filter_annotation 过滤掉了， 怎么办？
                # 因为在原来的代码中， 是不用获取 filename_g 的 target的
                # 先看一下 self.annos 里每个 video 被过滤的情况
                img_refs_g.append((img_ref, target_ref))

        target = self.get_groundtruth(idx)  # target is BoxList

        self.check_info(deepcopy(target.bbox), video_name, frame_id-1)

        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"bef",target)

        target = target.clip_to_image(remove_empty=True)

        # for box in target.bbox:
        #     print(box.to(torch.int64).tolist(),"aft",target)
        if self.transforms is not None:
            # 对于 target的 transform， 是根据对图片的resize来做target的resize
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i] = self.transforms(img_refs_l[i][0], img_refs_l[i][1])
            for i in range(len(img_refs_g)):
                img_refs_g[i] = self.transforms(img_refs_g[i][0], img_refs_g[i][1])
        else:
            assert False, "should use transforms"


        images = {}
        images["cur"] = (img, target)  # img 是tensor ， transforms 的最后有to_tensor 的操作
        images["ref_l"] = img_refs_l  # list[tuple], each one is (img_ref,target_ref)
        images["ref_g"] = img_refs_g  # list[tuple], each one is (img_g,target_g)
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["pattern"] = self.pattern[idx]  # # e.g., 0000_2401075277/%06d
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms
        images["filename"] = filename
        return images, idx

    # safe
    def get_groundtruth(self, idx):
        info = deepcopy(self.infos[idx])
        width, height = info["im_info"]  # NOTE im_info is w,h
        target = BoxList(info["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", info["labels"])
        target.add_field("tids", info["tids"])
        return target

    # safe
    def get_img_info(self, idx):
        im_info = deepcopy(self.infos[idx]["im_info"])
        return {"width": im_info[0], "height": im_info[1]}
        
        