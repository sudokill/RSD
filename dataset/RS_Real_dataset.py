from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from functools import partial
import torch
import numpy as np
from torch import nn
from PIL import Image


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def exists(x):
    return x is not None


    
    
class RS_Real_test_dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['npy', 'npz'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize((image_size,image_size)),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            #T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.transform_img = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            path = str(self.paths[index])
            
            # This is the line that was failing
            data_npz = np.load(path, allow_pickle=True)
            
            # If np.load succeeds, proceed as normal
            rs = data_npz['condition']
            out_flow = data_npz['out_flow']
            gs = data_npz['gs']
            out_flow = torch.from_numpy(out_flow).float()
            
            gs = Image.fromarray(gs)
            rs = Image.fromarray(rs)
            rs = self.transform(rs)
            gs = self.transform(gs)

            return [out_flow, rs, gs]
            
        except (EOFError, FileNotFoundError, OSError) as e:
            # If ANY error occurs while loading the file:
            print(f"\n[WARNING] Corrupted or missing file at index {index}: {self.paths[index]}. Error: {e}. Skipping and loading next item.")
            
            # Recursively call __getitem__ to get the next valid item in the list
            return self.__getitem__((index + 1) % len(self.paths))


   
    
class RS_Real_Train_dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['npy', 'npz'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print("Train image paths:", self.paths)
        print("Number of images loaded:", len(self.paths))

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize((image_size,image_size)),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            #T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.transform1 = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = str(self.paths[index])
        #path1 = str(path)
        
        data_npz = np.load(path, allow_pickle=True)
        rs = data_npz['condition']
        out_flow = data_npz['out_flow']
        gs = data_npz['gs']
        #out_flow = flow.transpose((2,0,1))
        #out_flow = torch.tensor(out_flow, dtype=torch.float32)
        out_flow = torch.from_numpy(out_flow).float()
        
        gs = Image.fromarray(gs)
        rs = Image.fromarray(rs)
        #rs = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
        rs = self.transform(rs)
        gs = self.transform(gs)

        path_out = Path(path).stem
        return [out_flow, rs, gs, rs.clone(), gs.clone(), path_out]