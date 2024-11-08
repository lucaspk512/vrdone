import os
import json

def sorted_listdir(dir):
    ls = os.listdir(dir)
    return sorted(ls)

if __name__ == '__main__':
    videos_dir = "./vidvrd/videos"
    frames_dir = "./vidvrd/frames"
    anno_dir = './vidvrd/annotations'
    splits = ['train', 'test']
    video_name_to_anno = dict()
    for st in splits:
        for video_name in os.listdir(os.path.join(anno_dir, st)):
            vid = video_name.split('.')[0]
            video_name_to_anno[vid] = os.path.join(anno_dir, st, video_name)
    
    video_names = sorted_listdir(videos_dir)
    for vn in video_names:
        vid = vn.split('.')[0]
        video_path = os.path.join(videos_dir, vn)

        fdir = os.path.join(frames_dir, vid)
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        
        with open(video_name_to_anno[vid], 'r') as f:
            anno = json.load(f)
        
        fps = anno['fps']
        output_name = os.path.join(fdir, vid+"_%06d.jpg")
        os.system('ffmpeg -i "{}" -r {} -q:v 1 "{}"'.format(video_path, fps, output_name))