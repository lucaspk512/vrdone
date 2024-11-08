import os
import json

def sorted_listdir(dir):
    ls = os.listdir(dir)
    return sorted(ls)

if __name__ == '__main__':
    videos_dir = "./vidor/videos"
    frames_dir = "./vidor/frames"
    anno_dirs = ['./vidor/annotations/training', './vidor/annotations/validation']
    
    video_groups = sorted_listdir(videos_dir)
    for vg in video_groups:
        video_names = sorted_listdir(os.path.join(videos_dir, vg))
        for vn in video_names:
            vid = vn.split('.')[0]
            video_path = os.path.join(videos_dir, vg, vn)

            fdir = os.path.join(frames_dir, vg, vid)
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            
            for anno_dir in anno_dirs:
                anno_path = os.path.join(anno_dir, vg, vid+'.json')
                if os.path.exists(anno_path):
                    with open(os.path.join(anno_dir, vg, vid+'.json'), 'r') as f:
                        anno = json.load(f)
                    
                    fps = anno['fps']
                    output_name = os.path.join(fdir, vid+"_%06d.jpg")
                    os.system('ffmpeg -i "{}" -r {} -q:v 1 "{}"'.format(video_path, fps, output_name))

                    break

