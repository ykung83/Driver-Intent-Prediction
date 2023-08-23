import os
from os.path import join
import copy

from multiprocessing import Pool
import tqdm

import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

from .dataset_utils import *

rng = np.random.default_rng(seed=42)
MAX_FRAMES=150


class B4CDataset(Dataset):
    """
    end_action - 0
    lchange - 1
    lturn - 2
    rchange - 3
    rturn - 4
    """
    def __init__(self, data_cfg, split="train", create_dataset=False):
        self.data_cfg   = data_cfg['DATALOADER_CONFIG']
        self.actions    = self.data_cfg['ACTIONS']
        self.cameras    = self.data_cfg['CAMERAS']
        self.data_dir   = self.data_cfg['DATA_DIR']
        self.split      = split

        videos_dict = self.read_videos_by_action()        

        if create_dataset:
            # self.create_gt_road_labels(videos_dict)
            self.generate_imagesets(videos_dict)
        
        self.image_sets = {}
        for camera in self.cameras:
            imageset_path = join(self.data_dir, f'ImageSets_{camera}', f'{self.split}.txt')
            self.image_sets[camera] = [line.strip() for line in open(imageset_path, 'r')]
        print(f'Added {len(self)} videos to the dataset.')

    def read_videos_by_action(self):
        videos_dict = {}
        # Combine image sets for all cameras
        for camera in self.cameras:
            camera_dir = join(self.data_dir, camera+"_processed")

            for action in self.actions:
                action_dir = join(camera_dir, action)

                if action not in videos_dict:
                    videos_dict[action] = []

                # Take the set intersection between all the cameras sequentially
                video_action_set = set([f for f in os.listdir(action_dir) if os.path.isdir(join(action_dir, f))])

                if len(videos_dict[action])==0:
                    videos_dict[action] = video_action_set
                else:
                    videos_dict[action] = videos_dict[action].intersection(video_action_set)

        # Convert videos_dict back to list
        for action in self.actions:
            videos_dict[action] = list(videos_dict[action])

        # Check that files exist
        for camera in self.cameras:
            camera_dir = join(self.data_dir, camera+"_processed")

            for action in self.actions:
                action_dir = join(camera_dir, action)

                for video_dir in videos_dict[action]:
                    video_dir = join(action_dir, video_dir)
                    assert os.path.exists(video_dir), f"Video directory {video_dir} does not exist."
        print("Finished reading all valid videos by directory.")
        return videos_dict

    @staticmethod
    def write_list_to_file(file_path, data_list):
        """
        Create/overwrite a .txt file and write each line of the Python list to a new line in the file.

        Parameters:
            file_path : str
                The path to the .txt file.
            data_list : list
                The Python list containing data to write to the file.
        """
        with open(file_path, 'w') as file:
            file.writelines(f"{item}\n" for item in data_list)

    def check_data_quality(self, video_subdirs, camera):
        """
        Check that all videos have the same number of frames and that the number of frames is less than MAX_FRAMES.

        Parameters:
            videos_dict : dict
                Dictionary containing the list of videos for each action.
        """

        valid_videos_mask = np.zeros(len(video_subdirs), dtype=bool)
        for video_idx, video_subdir in enumerate(video_subdirs):
            video_path = join(self.data_dir, video_subdir)
            data_dict = {}

            # Check that all videos have the full frame set
            if camera=="face":
                data_dict['gt_gazepose'] = self.get_face_labels(video_path)
                valid_videos_mask[video_idx] = len(data_dict['gt_gazepose'])>=MAX_FRAMES-1 # gazepose has 149
            elif camera=="road":
                data_dict['gt_bbox'], data_dict['gt_lanes'] = self.get_road_labels(video_path)
                valid_videos_mask[video_idx] = len(data_dict['gt_bbox'])>=MAX_FRAMES and len(data_dict['gt_lanes'])>=MAX_FRAMES

            if valid_videos_mask[video_idx]==0:
                print(f'Video {video_subdir} does not have the full frame set, skipping...')

        return valid_videos_mask
            

    def generate_imagesets(self, videos_dict):    
        print("Generating imagesets...")
        facecam_imageset_dict = {'train': [], 'val': [], 'test': []}
        roadcam_imageset_dict = copy.deepcopy(facecam_imageset_dict)
        train_pct, val_pct, test_pct = 0.7, 0.15, 0.15
        for action, action_videos in videos_dict.items():
            road_cam_action_dir = join('road_camera_processed_combined', action)
            road_cam_video_labels = np.array([join(road_cam_action_dir, f) for f in action_videos])
            road_cam_video_mask = self.check_data_quality(road_cam_video_labels, "road")

            face_cam_action_dir = join('face_camera_processed', action)
            face_cam_video_labels = np.array([join(face_cam_action_dir, f) for f in action_videos])
            face_cam_video_mask = self.check_data_quality(face_cam_video_labels, "face")

            combined_cam_video_mask = np.logical_and(road_cam_video_mask, face_cam_video_mask)
            road_cam_video_labels = road_cam_video_labels[combined_cam_video_mask]
            face_cam_video_labels = face_cam_video_labels[combined_cam_video_mask]

            road_cam_video_labels_sort_idx = np.argsort(road_cam_video_labels)
            road_cam_video_labels = road_cam_video_labels[road_cam_video_labels_sort_idx]
            face_cam_video_labels = face_cam_video_labels[road_cam_video_labels_sort_idx]

            # Ensure file order matches for road and face camera
            for i in range(len(road_cam_video_labels)):
                assert os.path.basename(road_cam_video_labels[i])==os.path.basename(face_cam_video_labels[i]), 'Video labels do not match'
            num_videos = len(road_cam_video_labels)

            num_train       = int(num_videos * train_pct)
            num_val         = int(num_videos * val_pct)
            num_test        = int(num_videos * test_pct)

            indices = np.arange(0, num_videos, 1)
            rng.shuffle(indices)

            videos_indices = {"train": [], "val": [], "test": []}
            videos_indices['train'], videos_indices['val'], videos_indices['test'] = indices[:num_train], \
                indices[num_train:num_train+num_val],  indices[num_train+num_val:num_train+num_val+num_test]

            for split in facecam_imageset_dict.keys():
                roadcam_imageset_dict[split].extend(road_cam_video_labels[videos_indices[split]].tolist())
                facecam_imageset_dict[split].extend(face_cam_video_labels[videos_indices[split]].tolist())

        # Dump to imageset files for road and face camera
        roadcam_imagesets_dir = join(self.data_dir, "ImageSets_road_camera")
        facecam_imagesets_dir = join(self.data_dir, "ImageSets_face_camera")
        if not os.path.exists(roadcam_imagesets_dir):
            print(f'Video root directory {roadcam_imagesets_dir} does not exist. Creating...')
            os.mkdir(roadcam_imagesets_dir)
        if not os.path.exists(facecam_imagesets_dir):
            print(f'Video root directory {facecam_imagesets_dir} does not exist. Creating...')
            os.mkdir(facecam_imagesets_dir)

        for split_key in facecam_imageset_dict.keys():
            road_cam_split_path = join(roadcam_imagesets_dir, f'{split_key}.txt')
            face_cam_split_path = join(facecam_imagesets_dir, f'{split_key}.txt')
            print(f'Saving imageset file {split_key} for road {road_cam_split_path} and {face_cam_split_path}')
            self.write_list_to_file(road_cam_split_path, roadcam_imageset_dict[split_key])
            self.write_list_to_file(face_cam_split_path, facecam_imageset_dict[split_key])

    def __len__(self):
        num_files = 0
        for camera in self.cameras: 
            assert num_files==0 or num_files==len(self.image_sets[camera]), "Number files in dataset not correct"
            num_files=len(self.image_sets[camera])
        return num_files
    
    def collate_fn(self, data):
        # print(data[0][1])
        data_batch = [bi[0] for bi in data]
        action_batch = [bi[1] for bi in data]
        return data_batch, action_batch

    def get_face_labels(self, label_dir):  
        gazepose_path = join(label_dir, 'gazepose.npy')
        assert os.path.exists(gazepose_path), f'Label file {gazepose_path} does not exist'
        gt_gazepose = np.load(gazepose_path)

        if gt_gazepose.shape[0] < MAX_FRAMES:
            # print(f'Gaze pose {gazepose_path} has less than 150 frames, padding with zeros...')
            # Pad with zeros
            gt_gazepose = np.pad(gt_gazepose, ((0, MAX_FRAMES-gt_gazepose.shape[0]), (0, 0)), mode='constant')
        # Assume gt_gazepose is not smaller than MAX_FRAMES
        num_frames = min(MAX_FRAMES, gt_gazepose.shape[0])
        gt_gazepose = gt_gazepose[:num_frames, :]

        return gt_gazepose

    def get_road_labels(self, label_dir):
        
        bbox_file = join(label_dir, 'bbox_labels.pkl')
        lane_file = join(label_dir, 'lane_labels.pkl')

        assert os.path.exists(bbox_file), f'Label directory {bbox_file} does not exist'
        assert os.path.exists(lane_file), f'Label directory {lane_file} does not exist'

        # Load bbox detections
        with open(bbox_file, 'rb') as f:
            gt_bbox = pickle.load(f)
        # Load road labels
        with open(lane_file, 'rb') as f:
            gt_lanes = pickle.load(f)

        gt_bbox = gt_bbox[:MAX_FRAMES]
        gt_lanes = gt_lanes[:MAX_FRAMES]

        return gt_bbox, gt_lanes
    
    def get_action_label(self, video_dir):
        action_label = video_dir.split('/')[-2]
        assert action_label in ACTION_TO_ID_MAP.keys(), f'Action {action_label} not in action map'
        action_id = ACTION_TO_ID_MAP[action_label]
        return action_id

    @staticmethod
    def combine_img_labels(args):
        split_video_path, combined_video_path = args
        img_label_files = [f for f in os.listdir(split_video_path) if f.endswith('.pkl')]

        MAX_NUM_BBOXES = 5 
        full_img_label_np = np.ones((MAX_FRAMES+1, MAX_NUM_BBOXES*5)) * -1
        num_img_label_files = len(img_label_files)
        for img_label_idx, img_label_file in enumerate(img_label_files):
            img_label_path = join(split_video_path, img_label_file)
            with open(img_label_path, 'rb') as f:
                img_data = pickle.load(f)
                IMG_W, IMG_H = 720, 480

                #1 Convert to xc, yc, w, h
                x1, x2, y1, y2 = img_data['xyxy'][:, 0], img_data['xyxy'][:, 2], img_data['xyxy'][:, 1], img_data['xyxy'][:, 3]
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                h = (x2 - x1)
                w = (y2 - y1)
            
                #2 Only keep bboxes with h and w < 0.33 (Ignore large bboxes of self)
                bbox_size_mask = np.logical_and(h < IMG_W*0.33, w < IMG_H*0.33)

                #3 Only keep labels with class_ids = 0, 1, 2, 3, 4 (Ignore 5 Date)
                class_ids = img_data['class_id'] 
                class_ids_mask = np.logical_and(class_ids >= 0, class_ids <= 4)

                bbox_mask = np.logical_and(bbox_size_mask, class_ids_mask)
                proc_img_label = np.ones((MAX_NUM_BBOXES, 5), dtype=int)*-1 # max of five bbox detections per image

                if np.sum(bbox_mask)>0:
                    xc = xc[bbox_mask].astype(int)
                    yc = yc[bbox_mask].astype(int)
                    h = h[bbox_mask].astype(int)
                    w = w[bbox_mask].astype(int)
                    class_ids = class_ids[bbox_mask]
                    gt_objs = np.stack((xc, yc, w, h, class_ids), axis=1)

                    #4 Select top 5 largest boxes
                    gt_obj_areas = gt_objs[:, 2] * gt_objs[:, 3]
                    gt_objs_sort_idx = np.argsort(-gt_obj_areas, kind='stable') # Sort high to low
                    num_objs = min(MAX_NUM_BBOXES, len(gt_objs_sort_idx))

                    proc_img_label[:num_objs, :] = gt_objs[gt_objs_sort_idx[:num_objs], :]

                proc_img_label = proc_img_label.flatten()
                proc_img_label = np.expand_dims(proc_img_label, axis=0)
                full_img_label_np[img_label_idx] = proc_img_label

        assert os.path.exists(combined_video_path), f'Video label directory {combined_video_path} does not exist'
        video_path = join(combined_video_path, 'bbox_labels.pkl')
        with open(video_path, 'wb') as f:
            pickle.dump(full_img_label_np, f)
        print("Saved combined bbox labels to: ", video_path)

        # Load, pad, save lane dets
        original_label_dir = split_video_path.replace('road_camera_processed', 'road_camera').replace('labels/', '')
        road_path = join(original_label_dir+".txt")
        gt_lanes = np.loadtxt(road_path, delimiter=',', dtype=int).reshape(1, -1)
        gt_lanes_padded = np.ones((MAX_FRAMES, 3)) * -1
        num_valid_gt_lanes = min(num_img_label_files, MAX_FRAMES)
        gt_lanes_padded[:num_valid_gt_lanes] = np.repeat(gt_lanes, [num_valid_gt_lanes], axis=0)

        lanes_path = join(combined_video_path, 'lane_labels.pkl')
        with open(lanes_path, 'wb') as f:
            pickle.dump(gt_lanes_padded, f)

    def create_gt_road_labels(self, videos_dict):
        split_video_path_list = []
        combined_video_path_list = []
        for action, action_videos in videos_dict.items():
            for video in action_videos:
                video_path = join(self.data_dir, 'road_camera_processed', 'labels', action, video)
                if not os.path.exists(video_path):
                    print(f'Video path {video_path} does not exist')
                    continue
                assert os.path.exists(video_path), f'Video path {video_path} does not exist'

                video_label_dir = join(self.data_dir, "road_camera_processed_combined", action, video)
                if not os.path.exists(video_label_dir):
                    print("Creating directory: ", video_label_dir)
                    os.makedirs(video_label_dir)
                
                split_video_path_list.append(video_path)
                combined_video_path_list.append(video_label_dir)
            # self.combine_img_labels((split_video_path_list[0], combined_video_path_list[0]))
        pool = Pool(processes=16)
        for _ in tqdm.tqdm(pool.imap_unordered(self.combine_img_labels, zip(split_video_path_list, 
            combined_video_path_list)), total=len(split_video_path_list)):
            pass

    def __getitem__(self, idx):
        data_dict = {}
        action_id=None

        for camera in self.cameras:
            video_subdir    = self.image_sets[camera][idx]
            video_fulldir   = join(self.data_dir, video_subdir)

            if camera == 'face_camera':
                data_dict['gt_gazepose'] = self.get_face_labels(video_fulldir)
            elif camera == 'road_camera':
                # Load all pickle files in the video directory
                data_dict['gt_bbox'], data_dict['gt_lanes'] = self.get_road_labels(video_fulldir)

            if action_id is None:
                action_id = self.get_action_label(video_fulldir)

        # Sort dict by key so that it is is consistent
        sorted_data_dict = dict(sorted(data_dict.items(), key=lambda item: item[0]))

        sorted_gt_np = np.empty((MAX_FRAMES, 0))
        # Stack all values from data_dict into a single np array

        for _, items in sorted_data_dict.items():
            sorted_gt_np = np.hstack((sorted_gt_np, items))

        # TOOD: Extract action label from Imageset file
        #  150 x [ (5x5) (2) (2) (3) ] # Pad if not enough frames obj_detections, gazepose, lane_detections # 150 x 32
        return sorted_gt_np, action_id # processed_input, action_label one hot vector 
