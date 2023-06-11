import os 
from PIL import Image
import torch
from torchvision import transforms


class Noise:
    def __init__(self, typeOfNoise="gaussian"):
        self.toTensor = transforms.ToTensor()
        self.toPILImage = transforms.ToPILImage()
        self.gaussianBlur = transforms.GaussianBlur(kernel_size=3)

        if typeOfNoise == "gaussian":
            self.noise = self.__add_gaussian_noise__

    def add_noise(self, src, dst):
        clean_files = os.listdir(src)
        
        for f in clean_files:
            with Image.open(src+f) as ip:
                new_img = self.toPILImage(self.noise(image=self.toTensor(ip)))
                new_img.save(dst+f)

    def __add_gaussian_noise__(self, image, mean=0.0, var=1):
        return self.gaussianBlur(image) #(image*255 + torch.randn(image.size()) * var**0.5 + mean)/255


noise = Noise()
noise.add_noise(src="./data/RainDrop/train/gt/", dst="./data/RainDrop/train/gauss_data/")