import os
import pickle
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from copy import deepcopy
import sys

import torch

from datasets.category import vidor_CatName2Id
from mega_core.structures.bounding_box import BoxList

class VidORDatasetGt(object):
    """ "this class is used to extract box_features of gt tracklets"""

    def __init__(
        self,
        cfg,
        image_set,
        data_dir,
        img_dir,
        anno_path,
        img_index,
        transforms,
        save_dir,
        part_id, 
    ):
        self.is_train = False
        self.cfg = cfg
        self.image_set = image_set  # e.g., VidORtrain_freq1_part01,
        self.transforms = transforms
        # test 的时候也是有 transform的，包括ColorJitter和像素的 Normalize，
        # 不然的话不适配网络的权重。所以直接之前写的把图片输入到backbone是不对的

        self.data_dir = data_dir  # data_dir == ../../"datasets"
        self.img_dir = img_dir  # img_dir: e.g.,  ../../datasets/vidor/frames
        self.anno_path = anno_path  # anno_path: e.g., ../../datasets/vidor/annotations/training
        self.img_index = img_index  # img_index: e.g., ../../datasets/vidor-dataset/img_index/VidOR_0001.txt
        self.save_dir = save_dir
        self.part_id = part_id

        # cache_dir = os.path.join(self.data_dir, 'VidOR_cache')
        # if not os.path.exists(cache_dir):
        #     os.mkdir(cache_dir)
        # self.cache_dir = cache_dir

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]  # .strip() 去除字符串首尾的空格

        # i.e., 对于视频的训练数据，是把视频提取为JPEG图片再做的
        self._img_dir = os.path.join(self.img_dir, "%s.jpg")  
        self._anno_path = os.path.join(self.anno_path, "%s/%s.json")  

        # e.g., 0000/2401075277/2401075277_000010   # 代表每张图片的名字
        # start from 1
        self.image_set_index = ["%s/%s/%s" % (x[0], x[1], '_'.join([x[1], "%06d"%(int(x[3]))])) for x in lines] 

        # e.g., 0000/2401075277/2401075277_%06d    # 代表每个video中图片名字的pattern
        self.pattern = [x[0] + '/' + x[1] + '/' + '_'.join([x[1], "%06d"]) for x in lines]

        self.frame_seg_id = [int(x[3]) for x in lines]
        self.frame_seg_len = [int(x[2]) for x in lines]

        # NOTE 我们现在是对gt跑test，所以还是要用到anno的
        keep = self.filter_annotation()

        self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
        self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
        self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self._prepare_video_frame_infos()
        self.Name2Id = vidor_CatName2Id

        # e.g., VidVRD_train_every10frames_anno.pkl
        self.annos = self.load_annos()  

        # self.annos is a list，其中每一个item 和 image_set_index中每个img是对应的。 顺序是保持对应的
        assert len(self.annos) == len(self.image_set_index)
        assert len(set(self.image_set_index)) == len(self.image_set_index)

        # # 在 self.load_annos 中用了 filter_annotation 之后的东西，但是在test的时候是没有filter_annotation的，
        # # 所以在 is_train == False 的时候， 就是说 train 和 test 用同一个数据集会出错，index out of range
        # self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_inference.pkl"))

        if not self.is_train:  # from vid_mega.py
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                temp = image_index.split("/")
                video_name, frame_id = '_'.join([temp[0], temp[1]]), temp[2].split('_')[1]
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
        
        # end_flag = True
        # for img_index in self.image_set_index:
        #     video_name = '_'.join(img_index.split('/')[:2])
        #     if not os.path.exists(os.path.join(self.save_dir, video_name+'.pkl')):
        #         end_flag = False
        
        # if end_flag:
        #     sys.exit()


    def __getitem__(self, idx):  # maintain
        # video_name = '_'.join(self.image_set_index[idx].split('/')[:2])
        # if os.path.exists(os.path.join(self.save_dir, video_name+'.pkl')):
        #     return None, idx

        return self._get_test(idx)

    def __len__(self):  # maintain
        return len(self.image_set_index)

    # safe
    def filter_annotation(self):  # 已改好
        # cache_file = os.path.join(self.cache_dir, self.image_set + "_keep_{}.pkl".format(self.part_id))

        # if os.path.exists(cache_file):
        #     print("Loading keep information from {} ... ".format(cache_file))
        #     with open(cache_file, "rb") as fid:
        #         keep = pickle.load(fid)
        #     print("Done.")
        #     return keep

        # self.image_set_index is before filtered
        video_name_dup_list = ['_'.join(x.split('/')[:2]) for x in self.image_set_index]  
        video_name_list = list(set(video_name_dup_list))
        video_name_list.sort(key=video_name_dup_list.index)  # keep the original order
        video_frame_count = {x: video_name_dup_list.count(x) for x in video_name_list}

        keep = np.zeros((len(self.image_set_index)), dtype=bool)
        outer_idx = 0
        # print("filtering annotations... ")

        # for video_name in tqdm(video_name_list):
        for video_name in video_name_list:
            with open(self._anno_path % tuple(video_name.split('_')), 'r') as json_file:
                video_ann = json.load(json_file)

            frame_count = video_frame_count[video_name]
            assert frame_count == int(video_ann['frame_count'])

            ### added in middle
            # gid, vn = video_name.split('_')
            # frame_count_extracted = len(os.listdir(os.path.join(self.img_dir, gid, vn)))
            # start_frame_name = os.path.join(self.img_dir, gid, vn, vn+'_{:06d}.jpg'.format(0))
            # end_frame_name = os.path.join(self.img_dir, gid, vn, vn+'_{:06d}.jpg'.format(frame_count_extracted-1))
            # assert os.path.exists(start_frame_name) and os.path.exists(end_frame_name)
            ###

            for frame_id in range(frame_count):
                objs = video_ann["trajectories"][frame_id]
                # keep[outer_idx] = False if (len(objs) == 0 or frame_id >= frame_count_extracted) else True
                keep[outer_idx] = False if len(objs) == 0 else True
                outer_idx += 1
        
        assert outer_idx == len(self.image_set_index)

        # with open(cache_file, "wb") as fid:
        #     pickle.dump(keep, fid)
        # print("keep information has been saved into {}".format(cache_file))
        return keep

    # safe
    def _prepare_video_frame_infos(self):
        # cache_file = os.path.join(self.cache_dir, self.image_set + "_frame_infos_{}.pkl".format(self.part_id))

        # if os.path.exists(cache_file):
        #     print("Loading keep information from {} ... ".format(cache_file))
        #     with open(cache_file, "rb") as fid:
        #         frame_infos = pickle.load(fid)
        #     (
        #         self.video_name_list,
        #         self.video_frame_count,
        #         self.filtered_frame_idx,
        #     ) = frame_infos
        #     print("Done.")
        #     return

        # print("preparing filtered video frame ids ...")
        # self.image_set_index is before filtered
        video_name_dup_list = ['_'.join(x.split('/')[:2]) for x in self.image_set_index]  
        self.video_name_list = list(set(video_name_dup_list))
        self.video_name_list.sort(key=video_name_dup_list.index)  # keep the original order
        self.video_frame_count = {x: video_name_dup_list.count(x) for x in self.video_name_list}

        self.filtered_frame_idx = {}

        # for video_name in tqdm(self.video_name_list):
        for video_name in self.video_name_list:
            # e.g., x == "0000/2401075277/2401075277_000010"
            temp = [x.split('/') for x in self.image_set_index] 
            frame_ids = sorted([int(x[2].split('_')[1]) for x in temp if '_'.join(x[:2]) == video_name])
            self.filtered_frame_idx[video_name] = frame_ids

        # frame_infos = (
        #     self.video_name_list,
        #     self.video_frame_count,
        #     self.filtered_frame_idx,
        # )
        # with open(cache_file, "wb") as fid:
        #     pickle.dump(frame_infos, fid)
        # print("frame_infos has been saved into {}".format(cache_file))

    # safe
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
    
    # safe
    def load_annos(self):
        # cache_file = os.path.join(self.cache_dir, self.image_set + "_anno_{}.pkl".format(self.part_id))
        # if os.path.exists(cache_file):
        #     print("loading annotation information from {} ... ".format(cache_file))
        #     with open(cache_file, "rb") as fid:
        #         annos = pickle.load(fid)
        #     print("Done.")
        #     return annos

        annos = []
        # print("construct annos... ")
        self.raw_annos = {}
        # for video_name in tqdm(self.video_name_list):
        for video_name in self.video_name_list:
            with open(self._anno_path % tuple(video_name.split('_')), 'r') as json_file:
                video_ann = json.load(json_file)
            
            self.raw_annos[video_name] = video_ann
            width_height = (int(video_ann["width"]), int(video_ann["height"]))
            
            # frame_ids = list(range(video_ann['frame_count']))
            # frame_ids = list(range(self.video_frame_count[video_name]))
            frame_ids = [fid - 1 for fid in self.filtered_frame_idx[video_name]]


            # tid 未必从 0 ~ len(traj_categories)-1 都有
            traj_categories = video_ann["subject/objects"]  
            # tid2category_map = [traj["category"] for traj in traj_categories] #  这样写是不对的, tid 未必从 0 ~ len(traj_categories)-1 都有
            tid2category_map = {traj["tid"]: traj["category"] for traj in traj_categories}  # 要这样搞
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'}

            for frame_id in frame_ids:
                objs = video_ann["trajectories"][frame_id]  # 长度为当前frame中bbox的数量
                anno = self._preprocess_annotation(objs, tid2category_map, width_height, video_name, frame_id)
                annos.append(anno) 

        # print("Saving annotation information into {} ... ".format(cache_file))
        # with open(cache_file, "wb") as fid:
        #     pickle.dump(annos, fid)
        # print("Done.")

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


    def _get_test(self, idx):  # 已好
        filename = self.image_set_index[idx]
        # e.g.,  "vidor/frames/%s.jpg"  %  "0001/10148360995/10148360995_000001"
        img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 for start, 1 for normal
        temp = filename.split("/")
        video_name, frame_id = '_'.join([temp[0], temp[1]]), temp[2].split('_')[1]
        
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
        img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
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
                img_ref = Image.open(self._img_dir % filename_g).convert("RGB")
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
