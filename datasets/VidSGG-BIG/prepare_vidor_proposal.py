import argparse
import pickle
import os
from tqdm import tqdm

from dataloaders.dataloader_vidor import TrajProposal

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare Data", add_help=False)
    parser.add_argument("--val_feature_path", type=str, default='../vidor/features/MEGA_VidORval_cache/MEGAv9_m60s0.3_freq1_VidORval_freq1_th_15-180-200-0.40.pkl')
    parser.add_argument("--save_path", type=str, default='../vidor/features/vidor_per_video_val')
    
    args = parser.parse_args()

    print("Loading validation features....")
    with open(args.val_feature_path, 'rb') as vf:
        mega_val_features = pickle.load(vf)
    
    print("Load done.")
    
    os.makedirs(args.save_path, exist_ok=True)

    for _key in tqdm(list(mega_val_features.keys())):
        proposal_dict = {k: v for k, v in mega_val_features[_key][0].__dict__.items()}
        data_dict = {"traj_proposal": proposal_dict}
        with open(os.path.join(args.save_path, _key+'.pkl'), 'wb') as sf:
            pickle.dump(data_dict, sf)

    print("Prepare done.")

