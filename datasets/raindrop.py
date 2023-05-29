import os
import zipfile
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class RainDrop(Dataset):
    def __init__(self, split) -> None:
        super().__init__()
        self.root_dir = "./data/RainDrop/"
        self.split = split
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=(200,500), antialias=True)

        if not os.path.exists(self.root_dir+ split):
            with zipfile.ZipFile(self.root_dir+ split + '.zip', 'r') as zipRef:
                zipRef.extractall(self.root_dir)
        
        self.noisy_files = os.listdir(self.root_dir + split + '/data/')
        self.GT = os.listdir(self.root_dir + split + '/gt/')
        
    def __len__(self):
        return len(os.listdir(self.root_dir + self.split + '/data/'))

    def __getitem__(self, index):
        src = self.root_dir + self.split

        with Image.open(src + '/data/' + self.noisy_files[index]) as img:
            data = self.resize(self.to_tensor(img))
        
        with Image.open(src + '/gt/' + self.GT[index]) as img:
            GT = self.resize(self.to_tensor(img))


        return data, GT
    
    def get_image_dimension(self):
        src = self.root_dir + self.split + '/data/'

        with Image.open(src + self.noisy_files[0]) as img:
            data = self.resize(self.to_tensor(img))
        
        return data.shape

    
