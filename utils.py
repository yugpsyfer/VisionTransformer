import os 
from PIL import Image
import torch
from torchvision import transforms
import numpy as np


class Noise:
    def __init__(self, typeOfNoise="gaussian"):
        self.toTensor = transforms.ToTensor()
        self.toPILImage = transforms.ToPILImage()
        self.gaussianBlur = transforms.GaussianBlur(kernel_size=15)

        if typeOfNoise == "gaussian":
            self.noise = self.__add_gaussian_noise__
        else:
            self.noise = self.__blackout_any_row_randomly__

    def add_noise(self, src, dst):
        clean_files = os.listdir(src)
        
        for f in clean_files:
            with Image.open(src+f) as ip:
                new_img = self.noise(image=ip)
                new_img.save(dst+f)

    def __add_gaussian_noise__(self, image, mean=1, var=16):
        return self.gaussianBlur(image) #(image*255 + torch.randn(image.size()) * var**0.5 + mean)/255
    
    def __add_salt_and_pepper_noise__(self, image):
        return  torch.tensor()
    
    def __blackout_any_row_randomly__(self, image):
        img_np = np.array(image)

        row_1 = np.random.randint(low=0, high=img_np.shape[1])
        row_2 = np.random.randint(low=0, high=img_np.shape[1])
        zeroes_per_channel = np.array([0 for _ in range(0,img_np.shape[2])])
        img_np[0,row_1:row_2,:] = zeroes_per_channel
        img_np[1,row_1:row_2,:] = zeroes_per_channel
        img_np[2,row_1:row_2,:] = zeroes_per_channel
        np_img = (img_np * 255).astype(np.uint8)

        return  Image.fromarray(np_img)


        

noise = Noise("s")
noise.add_noise(src="data/train/GT/", dst="data/train/corrupted_data/")