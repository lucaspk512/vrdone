from PIL import Image
from collections import deque

import torch

from mega_core.structures.image_list import to_image_list
from mega_core.structures.boxlist_ops import cat_boxlist
from mega_core.structures.bounding_box import BoxList
from mega_core.modeling.backbone.backbone import build_backbone
from mega_core.modeling.roi_heads.roi_heads import build_roi_heads

class FeatureExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super(FeatureExtractor, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.backbone = build_backbone(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.memory_enable = cfg.MODEL.VID.MEGA.MEMORY.ENABLE
        self.global_enable = cfg.MODEL.VID.MEGA.GLOBAL.ENABLE

        self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
        self.advanced_num = int(self.base_num * cfg.MODEL.VID.MEGA.RATIO)

        self.all_frame_interval = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL  # 25
        self.key_frame_location = cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION  # 12

        self.feats = None
        self.proposals = None
        self.proposals_dis = None
        self.proposals_feat = None
        self.proposals_feat_dis = None

        self.image_set_index = None 
        self.annos = None
        self.filtered_frame_idx = None

    def get_data_infos(self, image_set_index, annos, filtered_frame_idx):
        del self.image_set_index
        del self.annos
        del self.filtered_frame_idx
        
        self.image_set_index = image_set_index  # list[str] each str is similar as '0000_2401075277/000010'   # 代表每张图片的名字
        self.annos = annos
        self.filtered_frame_idx = filtered_frame_idx

    def get_groundtruth(self, filename):
        idx = self.image_set_index.index(filename)
        anno = self.annos[idx]

        width, height = anno["im_info"]  # NOTE im_info is w,h
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("tids", anno["tids"])

        return target


    def forward(self, images):
        """
        Arguments:
            images = {}
            images["cur"] = (img,target)    # img 是tensor ， transforms 的最后有to_tensor 的操作
            images["ref_l"] = img_refs_l    # list[tuple], each one is (img_ref,target_ref)
            images["ref_g"] = img_refs_g    # list[tuple], each one is (img_g,target_g)
            images["frame_category"] = frame_category
            images["seg_len"] = self.frame_seg_len[idx]
            images["pattern"] = self.pattern[idx]       # # e.g., 0000_2401075277/%06d
            images["img_dir"] = self._img_dir
            images["transforms"] = self.transforms
            images["filename"] = filename
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        assert not self.training

        images["cur"] = (to_image_list(images["cur"][0]), images["cur"][1])
        images["ref_l"] = [(to_image_list(img), tgt) for img, tgt in images["ref_l"]]
        images["ref_g"] = [(to_image_list(img), tgt) for img, tgt in images["ref_g"]]

        infos = images.copy()
        infos.pop("cur")

        return self._forward_test(images["cur"], infos)

    def _forward_test(self, img_tgt, infos):
        """
        forward for the test phase.
        :param img_tgt: (img,target)
        :param infos:
        :param targets:
        :return:
        """

        def update_feature(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (
                feats is not None
                and proposals is not None
                and proposals_feat is not None
            )
            assert (
                proposals != None
            ), "please input gt target as proposals to extract gt features"
            if img is not None:
                feats = self.backbone(img)[0]
                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                # proposals = self.rpn(imgs, (feats,), version="ref")  # rpn 返回的是一个 list， list of BoxList, len == batch_size

                proposals_feat = self.roi_heads.box.feature_extractor(
                    feats, proposals, pre_calculate=True
                )

            self.feats.append(feats)
            self.proposals.append(proposals[0])
            self.proposals_dis.append(
                proposals[0][: self.advanced_num]
            )  # BoxList 的对象是可以这样取值的， 有重载 __getitem__ 函数， 这里就是取前几个
            self.proposals_feat.append(proposals_feat)
            self.proposals_feat_dis.append(proposals_feat[: self.advanced_num])

        if infos["frame_category"] == 0:  # a new video
            # self.seg_len = infos["seg_len"]  # 这个是经过 dataset 里的 filter_annotation 之后的
            self.end_id = 0
            self.video_name = infos["filename"].split('/')[0]
            self.frame_ids = self.filtered_frame_idx[self.video_name]

            del self.feats
            del self.proposals
            del self.proposals_dis
            del self.proposals_feat
            del self.proposals_feat_dis

            self.feats = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)

            self.roi_heads.box.feature_extractor.init_memory()
            if self.global_enable:
                self.roi_heads.box.feature_extractor.init_global()

            img_cur, tgt_cur = img_tgt
            
            # ResNet-C4 只有一个 feature-level
            feats_cur = self.backbone(img_cur.tensors)[0]  
            # proposals_cur = self.rpn(imgs, (feats_cur, ), version="ref") # rpn 返回的是一个 list， list of BoxList, len == batch_size
            proposals_cur = [tgt_cur]
            proposals_feat_cur = self.roi_heads.box.feature_extractor(
                feats_cur, proposals_cur, pre_calculate=True
            )
            while len(self.feats) < self.key_frame_location + 1:
                update_feature(None, feats_cur, proposals_cur, proposals_feat_cur)

            while len(self.feats) < self.all_frame_interval:
                self.end_id = min(self.end_id + 1, len(self.frame_ids) - 1)
                end_filename = infos["pattern"] % self.frame_ids[self.end_id]
                end_image_ = Image.open(infos["img_dir"] % tuple(end_filename.split('/'))).convert("RGB")
                end_target = self.get_groundtruth(end_filename)
                # Problem: 如果 end_filename 不在 self.image_set_index 中怎么办, i.e, 它被 filter_annotation 过滤掉了， 怎么办？
                # 把 self.filtered_frame_idx 传进来

                # transforms 有 to tensor 的操作
                end_image, end_target = infos["transforms"](end_image_, end_target)  
                end_image_.close()
                
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(img_cur.tensors.device)
                end_target = end_target.to(end_image.device)
                update_feature(end_image, proposals=[end_target])

        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, len(self.frame_ids) - 1)
            # ref_l 里面直接取出来的已经经过 transforms 了, 这里的[0]因为它是一个len==1的list
            end_image, end_target = infos["ref_l"][0]  
            # end_image 是一个 ImageList 的对象
            end_image = end_image.tensors
            update_feature(end_image, proposals=[end_target])

        # 1. update global
        if infos["ref_g"]:
            for global_img, global_tgt in infos["ref_g"]:
                feats = self.backbone(global_img.tensors)[0]
                proposals = [global_tgt]
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)
                self.roi_heads.box.feature_extractor.update_global(proposals_feat)

        feats = self.feats[self.key_frame_location]

        img_cur, tgt_cur = img_tgt
        num_box = tgt_cur.bbox.shape[0]
        # proposals, proposal_losses = self.rpn(imgs, (feats, ), None)
        proposals = [tgt_cur]

        proposals_ref = cat_boxlist(list(self.proposals))
        proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
        proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
        proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

        proposals_list = [
            proposals,
            proposals_ref,
            proposals_ref_dis,
            proposals_feat_ref,
            proposals_feat_ref_dis,
        ]
        
        if self.roi_heads:
            # x, result, detector_losses = self.roi_heads(feats, proposals_list, None) # 原来是这样的
            # 但是现在我们不是过整个 roi_heads， 而是只需要其中的 feature_extractor
            box_features = self.roi_heads.box.feature_extractor(feats, proposals_list)
            assert num_box == box_features.shape[0], " box_features.shape = {}".format(box_features.shape)
        else:
            assert False, "please set roi_heads"

        return box_features


    def forward_second(self, images):
        """
        Arguments:
            images = {}
            images["cur"] = (img,target)    # img 是tensor ， transforms 的最后有to_tensor 的操作
            images["ref_l"] = img_refs_l    # list[tuple], each one is (img_ref,target_ref)
            images["ref_g"] = img_refs_g    # list[tuple], each one is (img_g,target_g)
            images["frame_category"] = frame_category
            images["seg_len"] = self.frame_seg_len[idx]
            images["pattern"] = self.pattern[idx]       # # e.g., 0000_2401075277/%06d
            images["img_dir"] = self._img_dir
            images["transforms"] = self.transforms
            images["filename"] = filename
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        assert not self.training

        for idx in range(len(images)):
            images[idx]["cur"] = (to_image_list(images[idx]["cur"][0]), images[idx]["cur"][1])
            images[idx]["ref_l"] = [(to_image_list(img), tgt) for img, tgt in images[idx]["ref_l"]]
            images[idx]["ref_g"] = [(to_image_list(img), tgt) for img, tgt in images[idx]["ref_g"]]

        infos = [img.copy() for img in images]
        
        for idx in range(len(infos)):
            infos[idx].pop("cur")

        return self._forward_test_second([img["cur"] for img in images], infos)

    def _forward_test_second(self, img_tgts, infos):
        """
        forward for the test phase.
        :param img_tgt: (img,target)
        :param infos:
        :param targets:
        :return:
        """
        def update_feature_second(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (
                feats is not None
                and proposals is not None
                and proposals_feat is not None
            )
            assert (
                proposals != None
            ), "please input gt target as proposals to extract gt features"
            if img is not None:
                feats = self.backbone(img)[0]
                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                # proposals = self.rpn(imgs, (feats,), version="ref")  # rpn 返回的是一个 list， list of BoxList, len == batch_size

                proposals_feat = self.roi_heads.box.feature_extractor(
                    feats, proposals, pre_calculate=True
                )
            
            self.feats.extend([feat[None, ...] for feat in feats])
            self.proposals.extend(proposals)
            self.proposals_dis.extend([proposals[idx][: self.advanced_num] for idx in range(len(proposals))])

            idx = 0
            for i in range(len(proposals)):
                self.proposals_feat.append(proposals_feat[idx: idx + len(proposals[i])])
                self.proposals_feat_dis.append(proposals_feat[idx: idx + min(len(proposals[i]), self.advanced_num)]) 

                idx += len(proposals[i])

        box_features_rst = []
        if 0 in [_info['frame_category'] for _info in infos]:
            zero_idx = -1
            for idx in range(len(infos)):
                if infos[idx]['frame_category'] == 0:
                    zero_idx = idx
                    break
            
            feats = []

            ### first
            if(zero_idx > 0):
                self.video_name = [_info["filename"].split('/')[0] for _info in infos[:zero_idx]]
                self.frame_ids = [self.filtered_frame_idx[vn] for vn in self.video_name]

                self.end_id = min(self.end_id + zero_idx, len(self.frame_ids) - 1)
                # ref_l 里面直接取出来的已经经过 transforms 了, 这里的[0]因为它是一个len==1的list
                
                end_images = [_info["ref_l"][0][0] for _info in infos[:zero_idx]]
                end_targets = [_info["ref_l"][0][1] for _info in infos[:zero_idx]]

                # end_image 是一个 ImageList 的对象
                end_images = torch.cat([end_img.tensors for end_img in end_images], dim=0)
                update_feature_second(end_images, proposals=end_targets)

                tgt_curs = [it[1] for it in img_tgts[:zero_idx]]
                proposals_ref = cat_boxlist(list(self.proposals))
                proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
                proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
                proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

                for idx in range(len(infos[:zero_idx])):
                    for global_img, global_tgt in infos[idx]["ref_g"]:
                        feats = self.backbone(global_img.tensors)[0]
                        proposals = [global_tgt]
                        proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)
                        self.roi_heads.box.feature_extractor.update_global(proposals_feat)

                    proposals_list = [
                        [tgt_curs[idx]],
                        proposals_ref,
                        proposals_ref_dis,
                        proposals_feat_ref,
                        proposals_feat_ref_dis,
                    ]
                    feats = self.feats[self.key_frame_location]
                    box_features = self.roi_heads.box.feature_extractor(feats, proposals_list)
                    box_features_rst.append(box_features)

            ### zero
            zero_img_tgt = img_tgts[zero_idx]
            zero_infos = infos[zero_idx]
            self.end_id = 0
            self.video_name = zero_infos["filename"].split('/')[0]
            self.frame_ids = self.filtered_frame_idx[self.video_name]

            del self.feats
            del self.proposals
            del self.proposals_dis
            del self.proposals_feat
            del self.proposals_feat_dis

            self.feats = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)
            self.roi_heads.box.feature_extractor.init_memory()
            if self.global_enable:
                self.roi_heads.box.feature_extractor.init_global()

            img_cur, tgt_cur = zero_img_tgt

             # ResNet-C4 只有一个 feature-level
            feats_cur = self.backbone(img_cur.tensors)[0]  
            # proposals_cur = self.rpn(imgs, (feats_cur, ), version="ref") # rpn 返回的是一个 list， list of BoxList, len == batch_size
            proposals_cur = [tgt_cur]
            proposals_feat_cur = self.roi_heads.box.feature_extractor(
                feats_cur, proposals_cur, pre_calculate=True
            )
            while len(self.feats) < self.key_frame_location + 1:
                update_feature_second(None, feats_cur, proposals_cur, proposals_feat_cur)

            while len(self.feats) < self.all_frame_interval:
                self.end_id = min(self.end_id + 1, len(self.frame_ids) - 1)
                end_filename = zero_infos["pattern"] % self.frame_ids[self.end_id]
                end_image_ = Image.open(zero_infos["img_dir"] % tuple(end_filename.split('/'))).convert("RGB")
                end_target = self.get_groundtruth(end_filename)
                # Problem: 如果 end_filename 不在 self.image_set_index 中怎么办, i.e, 它被 filter_annotation 过滤掉了， 怎么办？
                # 把 self.filtered_frame_idx 传进来

                # transforms 有 to tensor 的操作
                end_image, end_target = zero_infos["transforms"](end_image_, end_target)  
                end_image_.close()
                
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(img_cur.tensors.device)
                end_target = end_target.to(end_image.device)
                update_feature_second(end_image, proposals=[end_target])
                    
            ### last
            if(zero_idx < len(infos) - 1):
                self.video_name = [_info["filename"].split('/')[0] for _info in infos[zero_idx+1:]]
                self.frame_ids = [self.filtered_frame_idx[vn] for vn in self.video_name]

                self.end_id = min(self.end_id + len(infos) - zero_idx - 1, len(self.frame_ids) - 1)
                # ref_l 里面直接取出来的已经经过 transforms 了, 这里的[0]因为它是一个len==1的list
                
                end_images = [_info["ref_l"][0][0] for _info in infos[zero_idx+1:]]
                end_targets = [_info["ref_l"][0][1] for _info in infos[zero_idx+1:]]

                # end_image 是一个 ImageList 的对象
                end_images = torch.cat([end_img.tensors for end_img in end_images], dim=0)
                update_feature_second(end_images, proposals=end_targets)
            
            tgt_curs = [it[1] for it in img_tgts[zero_idx:]]
            proposals_ref = cat_boxlist(list(self.proposals))
            proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
            print([self.proposals_feat[i].shape for i in range(len(self.proposals_feat))])
            proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
            proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

            for idx in range(len(infos[zero_idx:])):
                for global_img, global_tgt in infos[idx]["ref_g"]:
                    _feats = self.backbone(global_img.tensors)[0]
                    proposals = [global_tgt]
                    proposals_feat = self.roi_heads.box.feature_extractor(_feats, proposals, pre_calculate=True)
                    self.roi_heads.box.feature_extractor.update_global(proposals_feat)

                proposals_list = [
                    [tgt_curs[idx]],
                    proposals_ref,
                    proposals_ref_dis,
                    proposals_feat_ref,
                    proposals_feat_ref_dis,
                ]
                feats = self.feats[self.key_frame_location]
                box_features = self.roi_heads.box.feature_extractor(feats, proposals_list)
                box_features_rst.append(box_features)

        else:
            self.video_name = [_info["filename"].split('/')[0] for _info in infos]
            self.frame_ids = [self.filtered_frame_idx[vn] for vn in self.video_name]

            self.end_id = min(self.end_id + len(infos), len(self.frame_ids) - 1)
            # ref_l 里面直接取出来的已经经过 transforms 了, 这里的[0]因为它是一个len==1的list
            
            end_images = [_info["ref_l"][0][0] for _info in infos]
            end_targets = [_info["ref_l"][0][1] for _info in infos]

            # end_image 是一个 ImageList 的对象
            end_images = torch.cat([end_img.tensors for end_img in end_images], dim=0)
            update_feature_second(end_images, proposals=end_targets)

            tgt_curs = [it[1] for it in img_tgts]
            proposals = tgt_curs
            proposals_ref = cat_boxlist(list(self.proposals))
            proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
            proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
            proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

            for idx in range(len(infos)):
                for global_img, global_tgt in infos[idx]["ref_g"]:
                    feats = self.backbone(global_img.tensors)[0]
                    proposals = [global_tgt]
                    proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)
                    self.roi_heads.box.feature_extractor.update_global(proposals_feat)

                feats = self.feats[self.key_frame_location]
                proposals_list = [
                    [tgt_curs[idx]],
                    proposals_ref,
                    proposals_ref_dis,
                    proposals_feat_ref,
                    proposals_feat_ref_dis,
                ]

                box_features = self.roi_heads.box.feature_extractor(feats, proposals_list)
                box_features_rst.append(box_features)

        return box_features_rst

