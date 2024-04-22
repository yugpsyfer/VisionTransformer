import os
import zipfile
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class RainDrop(Dataset):
    def __init__(self, split, root_dir="./data/") -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=(128,128), antialias=True)

        # if not os.path.exists(self.root_dir+ split):
        #     with zipfile.ZipFile(self.root_dir+ split + '.zip', 'r') as zipRef:
        #         zipRef.extractall(self.root_dir)
        
        # if not os.path.exists(self.root_dir + '/gauss_data/'):
        #     os.mkdir(self.root_dir + '/gauss_data/')

        self.noisy_files = os.listdir(self.root_dir + split + '/gauss_data/')  # replaced /data/ with /gt/
        self.GT = os.listdir(self.root_dir + split + '/GT/')
        
    def __len__(self):
        return len(os.listdir(self.root_dir + self.split + '/gauss_data/'))

    def __getitem__(self, index):
        src = self.root_dir + self.split

        with Image.open(src + '/gauss_data/' + self.noisy_files[index]) as img:   # replaced /data/ with /gt/
            data = self.resize(self.to_tensor(img))/255
        
        with Image.open(src + '/GT/' + self.GT[index]) as img:
            GT = self.resize(self.to_tensor(img))/255


        return data, GT
    
    def get_image_dimension(self):
        src = self.root_dir + self.split + '/GT/'

        with Image.open(src + self.GT[0]) as img:
            data = self.resize(self.to_tensor(img))
        
        return data.shape

    
