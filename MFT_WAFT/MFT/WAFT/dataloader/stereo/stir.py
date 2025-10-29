import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os.path as osp


class STIR(Dataset):
    #def __init__(self, root='/Datasets/STIRDataset', split='3', transform=None):
    #    self.root = root
    #    self.split = split
    #    self.transform = transform or transforms.ToTensor()
    #    self.samples = []

    #    self._gather_sequences()


    def __init__(self, aug_params=None, split='training', transform=None, 
                 root='/Datasets/STIRDataset/'):#, resize_or_crop='resize', target_size=(432, 960)):        
        
        super(STIR, self).__init__()

        self.aug_params = aug_params
        self.root = root
        self.split = split
        self.transform = transform or (lambda x: torch.from_numpy(np.array(x)))
        #self.resize_or_crop = resize_or_crop
        #self.target_size = target_size
        #self.samples = []
        
        #if resize_or_crop == 'resize':
        #    self.frame_transform = transforms.Compose([
        #        transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
        #        transforms.ToTensor()
        #    ])
        #    self.mask_transform = transforms.Compose([
        #        transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST)
        #    ])
        #elif resize_or_crop == 'center_crop':
        #    self.frame_transform = transforms.Compose([
        #        transforms.CenterCrop(target_size),
        #        transforms.ToTensor()
        #    ])
        #    self.mask_transform = transforms.Compose([
        #        transforms.CenterCrop(target_size)
        #    ])
        #elif resize_or_crop == 'random_crop':
        #    self.frame_transform = transforms.Compose([
        #        transforms.RandomCrop(target_size),
        #        transforms.ToTensor()
        #    ])
        #    self.mask_transform = transforms.Compose([
        #        transforms.RandomCrop(target_size)
        #    ])
        #else:
        #    raise ValueError(f"Invalid resize_or_crop: {resize_or_crop}")

        # Gather scene list
        scenes = sorted(
            [f for f in glob.glob(osp.join(root, '*')) if os.path.basename(f).isdigit()],
            key=lambda x: int(os.path.basename(x))
        )
        split_file = osp.join(root, 'stir_split.txt')
        split_list = np.loadtxt(split_file, dtype=np.int32)

        assert len(scenes) == len(split_list), \
            f"Expected {len(split_list)} scenes from split file, but found {len(scenes)} folders."

        self.scene_list = [
            scene for scene, split_id in zip(scenes, split_list)
            if (split == 'training' and split_id == 1) or (split == 'validation' and split_id == 2)
        ]

        self.samples = self._gather_sequences()


    def _gather_sequences(self):
        samples = []
        for scene_path in self.scene_list:
            left_path = os.path.join(scene_path, 'left')
            if not os.path.exists(left_path):
                continue

            seq_folders = sorted(os.listdir(left_path))
            #print(left_path)
            for seq in seq_folders:
                seq_path = os.path.join(left_path, seq)

                frame_folder = os.path.join(seq_path, 'frames')
                segmentation_folder = os.path.join(seq_path, 'segmentation')

                video_files = glob.glob(os.path.join(frame_folder, '*.mp4'))
                if not video_files:
                    continue

                video_path = video_files[0]
                seg_start = os.path.join(segmentation_folder, 'icgstartseg.png')
                seg_end = os.path.join(segmentation_folder, 'icgendseg.png')

                if os.path.exists(seg_start) and os.path.exists(seg_end):
                    samples.append({
                        'video': video_path,
                        'seg_start': seg_start,
                        'seg_end': seg_end
                    })
        #print(samples[0])
        return samples



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample = self.samples[index]

        # --- Read video frames ---
        cap = cv2.VideoCapture(sample['video'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 2:
            raise RuntimeError(f"Video {sample['video']} has less than 2 frames!")

        frames = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {frame_idx} from {sample['video']}")
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frames.append(Image.fromarray(frame))  
            #frame = Image.fromarray(frame)
            #frame_tensor = self.frame_transform(frame)  
            #frames.append(frame_tensor)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # still a NumPy array
            frame = Image.fromarray(frame)                   # convert to PIL
            frame_tensor = torch.from_numpy(np.array(frame)).permute(2,0,1).float()       
            frames.append(frame_tensor)

        cap.release()
        # --- Load masks ---
        seg_start = Image.open(sample['seg_start']).convert("L")
        seg_end = Image.open(sample['seg_end']).convert("L")
        
        seg_start = torch.from_numpy(np.array(seg_start)).long()  # [H,W], no channel
        seg_end   = torch.from_numpy(np.array(seg_end)).long()

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        return frames, seg_start, seg_end
    
        ## --- Load masks as PIL images ---
        #seg_start = Image.open(sample['seg_start']).convert("L")
        #seg_end   = Image.open(sample['seg_end']).convert("L")  #
        ## --- Resize transforms ---
        #frame_transform = transforms.Compose([
        #    transforms.Resize((432, 960), interpolation=InterpolationMode.BILINEAR),
        #    transforms.ToTensor()
        #])
        #mask_transform = transforms.Resize((432, 960), interpolation=InterpolationMode.NEAREST) #
        ## Apply to frames
        #frames = [frame_transform(f) for f in frames]   #
        ## Apply to masks
        #seg_start = mask_transform(seg_start)
        #seg_end   = mask_transform(seg_end) #
        #seg_start = torch.from_numpy(np.array(seg_start)).long()
        #seg_end   = torch.from_numpy(np.array(seg_end)).long()  #
        ## Stack frames into tensor [T, C, H, W]
        #frames = torch.stack(frames, dim=0) #
        #return frames, seg_start, seg_end


#    def __getitem__(self, index):
#        sample = self.samples[index]
#
#        # Open the video
#        cap = cv2.VideoCapture(sample['video'])
#        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#        if total_frames < 2:
#            raise RuntimeError(f"Video {sample['video']} has less than 2 frames!")
#
#        frames = []
#
#        for frame_idx in range(total_frames):
#            ret, frame = cap.read()
#            if not ret:
#                raise RuntimeError(f"Could not read frame {frame_idx} from {sample['video']}")
#
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#            frame = self.transform(Image.fromarray(frame))
#            frames.append(frame)
#
#        cap.release()
#
#        # Load start and end segmentations (first and last)
#        seg_start = np.array(Image.open(sample['seg_start']))
#        seg_end = np.array(Image.open(sample['seg_end']))
#
#        seg_start = torch.from_numpy(seg_start).long()
#        seg_end = torch.from_numpy(seg_end).long()
#
#        frames = torch.stack(frames, dim=0)
#        
#        return frames, seg_start, seg_end

    #def __getitem__(self, index):
    #    sample = self.samples[index]
#
    #    # --- Read video frames ---
    #    cap = cv2.VideoCapture(sample['video'])
    #    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
    #    if total_frames < 2:
    #        raise RuntimeError(f"Video {sample['video']} has less than 2 frames!")
#
    #    frames = []
    #    for frame_idx in range(total_frames):
    #        ret, frame = cap.read()
    #        if not ret:
    #            raise RuntimeError(f"Could not read frame {frame_idx} from {sample['video']}")
    #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        frame = Image.fromarray(frame)
    #        frames.append(frame)  # Keep as PIL for joint transform later
#
    #    cap.release()
#
    #    # --- Load masks as PIL images ---
    #    seg_start = Image.open(sample['seg_start']).convert("L")  # Single channel
    #    seg_end = Image.open(sample['seg_end']).convert("L")
#
    #    # --- Apply same spatial transform to frames + masks ---
    #    if self.transform:
    #        # Apply to frames
    #        frames = [self.transform(f) for f in frames]
#
    #        # For masks, apply same transform but without normalisation
    #        mask_transform = transforms.Compose([
    #            transforms.Resize(frames[0].shape[1:], InterpolationMode.BILINEAR)  # match H,W
    #        ])
    #        seg_start = mask_transform(seg_start)
    #        seg_end = mask_transform(seg_end)
#
    #        seg_start = torch.from_numpy(np.array(seg_start)).long()
    #        seg_end = torch.from_numpy(np.array(seg_end)).long()
    #    else:
    #        # Default to tensor conversion
    #        frames = [transforms.ToTensor()(f) for f in frames]
    #        seg_start = torch.from_numpy(np.array(seg_start)).long()
    #        seg_end = torch.from_numpy(np.array(seg_end)).long()
#
    #    # Stack frames into tensor [T, C, H, W]
    #    frames = torch.stack(frames, dim=0)
#
    #    return frames, seg_start, seg_end
#



            #split_path = os.path.join(self.root, self.split, 'left')
            #sequences = sorted(os.listdir(split_path))
            #frame_folder = os.path.join(split_path, seq, 'frames')
            #segmentation_folder = os.path.join(split_path, seq, 'segmentation')

            ## Find video file inside 'frames' folder
            #video_files = glob.glob(os.path.join(frame_folder, '*.mp4'))
            #if not video_files:
            #    continue

            #video_path = video_files[0]

            ## Check for segmentation masks
            #seg_start = os.path.join(segmentation_folder, 'icgstartseg.png')
            #seg_end = os.path.join(segmentation_folder, 'icgendseg.png')

            #if os.path.exists(seg_start) and os.path.exists(seg_end):
            #    self.samples.append({
            #        'video': video_path,
            #        'seg_start': seg_start,
            #        'seg_end': seg_end
            #    })



            
        #self.frame_transform = transforms.Compose([
        #    transforms.Resize((432, 960), interpolation=InterpolationMode.BILINEAR),
        #    transforms.ToTensor()
        #])
        #self.frame_transform = transforms.Compose([
        #    transforms.CenterCrop((432, 960)),
        #    transforms.ToTensor()
        #])
        #self.mask_transform = transforms.Compose([
        #    transforms.Resize((432, 960), interpolation=InterpolationMode.NEAREST)
        #])
        
        #scenes = sorted(glob.glob(osp.join(root, '*')), key=lambda x: int(os.path.basename(x)))

        #scenes = sorted(
        #    [f for f in glob.glob(osp.join(root, '*')) if os.path.basename(f).isdigit()],
        #    key=lambda x: int(os.path.basename(x))
        #)
        #split_file = osp.join(root, 'stir_split.txt')
        #split_list = np.loadtxt(split_file, dtype=np.int32)

        #assert len(scenes) == len(split_list), \
        #    f"Expected {len(split_list)} scenes from split file, but found {len(scenes)} folders."

        #self.scene_list = [
        #    scene for scene, split_id in zip(scenes, split_list)
        #    if (split == 'training' and split_id == 1) or (split == 'validation' and split_id == 2)
        #]

        ## Gather sequences (list of frame paths)
        #self.samples = self._gather_sequences()