import os
import argparse
import pickle
from tqdm import tqdm
from copy import deepcopy

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=1000)

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

from vidvrd_dataset import VidVRDDatasetProposal
from feature_extractor_vidvrd import FeatureExtractor


BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

def extract_VidVRD_test_features(gpu_id, config_file, args):
    from mega_core.config import cfg
    from mega_core.data.transforms.build import build_transforms
    from mega_core.utils.checkpoint import DetectronCheckpointer
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(config_file)

    dataset = VidVRDDatasetProposal(
        cfg=cfg,
        img_dir=args.frame_dir,
        proposal_path=args.proposal_path,
        anno_path=args.anno_dir,
        transforms=build_transforms(cfg, is_train=False),
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        drop_last=False, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=lambda x: x,
        worker_init_fn=set_worker_sharing_strategy
    )


    device = torch.device("cuda:{}".format(gpu_id))
    model = FeatureExtractor(cfg)
    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, use_latest=False, flownet=None)
    model = model.to(device)
    model.eval()

    model.get_data_infos(deepcopy(dataset.image_set_index), deepcopy(dataset.infos), deepcopy(dataset.filtered_frame_idx))

    video_frame_count = deepcopy(dataset.video_frame_count)
    video_gt_features = dict()
    
    video_all_index = 0

    for data in tqdm(dataloader):
        images, _ = data[0]

        # if images == None:
        #     continue
        
        fname_part = images["filename"].split("/")
        video_name, frame_id = fname_part[0], fname_part[1].split('_')[-1]
        feature_dict_path = os.path.join(save_dir, video_name + '.pkl')

        frame_id = int(frame_id)
        img, target = images["cur"]
        tids = target.get_field("tids").cpu().numpy()

        video_gt_features[frame_id] = {
            "frame_id": frame_id,
            "tids": tids
        }
        images["cur"] = (img.to(device), target.to(device))
        for key in ("ref", "ref_l", "ref_m", "ref_g"):
            if key in images.keys():
                images[key] = [
                    (img.to(device), target.to(device)) for img, target in images[key]
                ]
        
        with torch.no_grad():
            box_features = model(images)

        video_gt_features[frame_id]['visual_features'] = box_features.cpu().numpy()

        video_frame_count[video_name] -= 1
        if video_frame_count[video_name] == 0:
            with open(feature_dict_path, 'wb') as ff:
                pickle.dump(video_gt_features, ff)
            print("Saved video {}th: {} done.".format(video_all_index, video_name))
            video_all_index += 1
            video_gt_features = dict()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='xxxx')
    parser.add_argument('--gpu_id', type=str, help='the dataset name for evaluation')
    parser.add_argument('--config_file', type=str, default="configs/MEGA/partxx/VidVRDtrain_freq5.yaml")
    parser.add_argument('--save_dir', type=str, default="../vidvrd/features/Proposal_boxfeatures_test")
    parser.add_argument('--frame_dir', type=str, default="../vidvrd/frames")
    parser.add_argument('--anno_dir', type=str, default="../vidvrd/annotations/test")
    parser.add_argument('--proposal_path', type=str, default="../vidvrd/features/vidvrd_per_video_val")
    parser.add_argument('--lr', type=str, default="1")
    parser.add_argument('--model_idx', type=int, default=190000)

    args = parser.parse_args()
    gpu_id = int(args.gpu_id)

    extract_VidVRD_test_features(gpu_id, args.config_file, args)
