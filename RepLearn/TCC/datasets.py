
import os
import random

import cv2
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.parser import parse_args, load_config
from utils.utils import _extract_frames_h5py, gen_labels


class VideoAlignmentLoader(Dataset):
    def __init__(self, cfg, get_annotation=False):
        self.cfg = cfg
        self.get_annotation = get_annotation
        videos_dir = self.cfg.VAOT.DATA_PATH
        if self.get_annotation:
            # Load the annotations
            category_id = videos_dir.split('/')[0]
            if len(category_id) ==  0:
                category_id = videos_dir.split('/')[-2]
            category_id_split = category_id.split('_')
            # For the cases when we split the data into train and test and
            # rename the folder as `48448_7150991_train`
            if len(category_id_split) == 2:
                pass
            else:
                assert len(category_id_split) == 3
                category_id = "_".join(category_id_split[:2])
            ann_dir = os.path.join(self.cfg.ANNOTATION.PATH, category_id)
            assert os.path.isdir(ann_dir)
            self.ann_paths = [
                os.path.join(ann_dir, item) for item in os.listdir(ann_dir)
            ]
        videos_names = os.listdir(videos_dir)
        self.video_paths = [
            os.path.join(videos_dir, item) for item in videos_names
        ]
        self.num_frames = self.cfg.VAOT.NUM_FRAMES
        self.num_context_steps = self.cfg.VAOT.NUM_CONTEXT_STEPS
        # Number of frames to sample per video (including the context frames)
        self.frames_per_video = self.num_frames * self.num_context_steps

        if cfg.DATA_LOADER.NAME == 'CMU_Kitchens':
            self.FRAMES_DIR = self.cfg.CMU_KITCHENS.FRAMES_PATH
        elif cfg.DATA_LOADER.NAME == 'EGTEA_GazeP':
            self.FRAMES_DIR = self.cfg.EGTEA_GAZEP.FRAMES_PATH
        elif cfg.DATA_LOADER.NAME == 'MECCANO':
            self.FRAMES_DIR = self.cfg.MECCANO.FRAMES_DIR
        elif cfg.DATA_LOADER.NAME == 'EPIC-Tents':
            self.FRAMES_DIR = self.cfg.TENTS.FRAMES_DIR
        elif cfg.DATA_LOADER.NAME == 'ProceL':
            self.FRAMES_DIR = self.cfg.PROCEL.FRAMES_PATH
        elif cfg.DATA_LOADER.NAME == 'CrossTask':
            self.FRAMES_DIR = self.cfg.CROSSTASK.FRAMES_PATH
        elif cfg.DATA_LOADER.NAME == 'pc_assembly':
            self.FRAMES_DIR = self.cfg.PCASSEMBLY.FRAMES_DIR
        elif cfg.DATA_LOADER.NAME == 'pc_disassembly':
            self.FRAMES_DIR = self.cfg.PCDISASSEMBLY.FRAMES_DIR
        else:
            raise NotImplementedError

    def __len__(self):
        return self.cfg.VAOT.BATCH_SIZE

    def __getitem__(self, idx):
        assert self.cfg.VAOT.BATCH_SIZE == 2    # YOU MUST KEEP IT AT 2 WHEN RUNNING VAOT
        assert self.cfg.VAOT.BATCH_SIZE < len(self.video_paths)
        # Randomly select a batch of videos from the available video paths
        selected_videos = random.sample(
            self.video_paths,
            self.cfg.VAOT.BATCH_SIZE
        )
        final_frames = list()
        seq_lens = list()
        steps = list()
        if self.get_annotation:
            annotations = list()
        # Iterate through the selected batch of videos
        for video in selected_videos:
            # Use CV2 to get the no. of frames in the video
            video_frames_count = self.get_num_frames(video)

            #NOTE: We don't need this if we used check_vid_for_opencv_errors(EPIC-Tents).sh to re-encode the videos
            #BUG: When using EPIC-Tents, this video was corrupt so it returns an incorrect framecount, we'd use the framecount of the h5 file instead
            #BUG: Ignore these warnings: Invalid NAL unit size, Error splitting the input into NAL units.
            # if video.split('/')[-1][:-4]=='02.tent.120617.gopro':
            #     video_frames_count = 41291

            # Sample frames from the video and return their frame indices: 
            # selected_frames includes the main and context frames, main_frames doesn't include the context frames
            main_frames, selected_frames = self.get_frame_sequences(
                video_frames_count
            )
            # Extract and resize ALL the frames from the video, store in a h5 file, and return its filename 
            h5_file_name = _extract_frames_h5py(
                video,
                self.FRAMES_DIR
            )
            # Read the saved h5 file and extract ONLY the sampled frames, resize and normalize, return the list of frames
            frames = self.get_frames_h5py(
                h5_file_name,
                selected_frames,
            )
            frames = np.array(frames)
            final_frames.append(
                np.expand_dims(frames.astype(np.float32), axis=0)
            )
            steps.append(np.expand_dims(np.array(main_frames), axis=0))
            seq_lens.append(video_frames_count)
            if self.get_annotation:
                ann = self.generate_annotation(video, video_frames_count)
                assert len(ann) == video_frames_count
                ann[ann == -1] = 0
                annotations.append(ann[main_frames])
        if self.get_annotation:
            return (
                np.concatenate(final_frames),
                np.concatenate(steps),
                np.array(seq_lens),
                np.vstack(annotations),
            )
        # return a tuple of 3 numpy arrays:
        # frames: (batch_size, num_sampled_frames[main+context], 168, 168, 3)
        # frame indices: (batch_size, num_sampled_frames[main])
        # no. of frames in the videos: (batch_size,)
        return (
            np.concatenate(final_frames),
            np.concatenate(steps),
            np.array(seq_lens),
        )

    def get_frames_h5py(self, h5_file_path, frames_list):
        final_frames = list()
        h5_file = h5py.File(h5_file_path, 'r')
        frames = h5_file['images']
        for frame_num in frames_list:
            frame_ = frames[frame_num]
            frame = cv2.resize(
                frame_,
                (168, 168),
                interpolation=cv2.INTER_AREA
            )
            frame = (frame / 127.5) - 1.0
            final_frames.append(frame)
        h5_file.close()
        assert len(final_frames) == len(frames_list)
        return final_frames

    def get_num_frames(self, video):
        """
        This method is used to calculate the number of frames in a video.
        """
        cap = cv2.VideoCapture(video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return num_frames

    def get_video_fps(self, video):
        """
        This method is used to calculate the fps of a video.
        """
        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def get_frame_sequences(self, video_frames_count):
        """
        This method is used generate the frame ids which are to be sampled.

        Args:
            video_frames_count (int): Number of frames in the video.
        """
        # Selecting the frames at random
        # BUG: This will throw an exception if the video is too short or if we sample a large no. of frames, both are unlikely for our project, the solution for this would be masks like in VAOT
        main_frames = sorted(
            random.sample(range(video_frames_count), self.num_frames)
        )
        # Generating corresponding context frames for those frames
        w_context_frames = list()
        for frame in main_frames:
            w_context_frames.append(frame)
            for i in range(self.num_context_steps - 1):
                context_frame_count = frame + (
                    (i + 1) * self.cfg.VAOT.CONTEXT_STRIDE
                )
                if context_frame_count >= video_frames_count - 1:
                    context_frame_count -= video_frames_count - 2
                w_context_frames.append(context_frame_count)
        assert len(w_context_frames) == self.frames_per_video
        assert max(w_context_frames) <= video_frames_count
        return main_frames, w_context_frames

    def generate_annotation(self, video, num_frames):
        """
        This method is used to generate the annotations for the videos.
        """
        file_name = video.split('/')[-1].split('.')[0]
        for item in self.ann_paths:
            if file_name in item:
                ann_file_name = item
        assert os.path.isfile(ann_file_name)
        annotation_data = pd.read_csv(open(ann_file_name, 'r'), header=None)
        fps = self.get_video_fps(video)
        ann = gen_labels(fps, annotation_data.values, num_frames)
        return ann


if __name__ == '__main__':
    cfg = load_config(parse_args())
    data_loader = DataLoader(
        VideoAlignmentLoader(cfg),
        batch_size=1,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS
    )
    batch_count = 0
    while True:
        batch_count += 1
        print(f'Batch {batch_count}; len {len(data_loader)}:')
        frames, steps, seq_lens = next(iter(data_loader))
        if batch_count == 100:
            break
