import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import io
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
import os

'''TRAIN_PATHS = './train_paths.txt'
TEST_PATHS = './test_paths.txt' '''

TRAIN_PATHS = 'BTS/dataset/train_paths.txt'
TEST_PATHS = 'BTS/dataset/test_paths.txt'

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class NYUDatasetRGBD(Dataset):
    def __init__(self, data_file, mode, isCAML = False, transform=None):

        self.focal_length = 518.8579 
        self.mode = mode
        #are we training in CAML?
        self.isCAML = isCAML

        
        with io.open(data_file, 'r') as file:
            self.file_paths = file.readlines() #stores each line in list


        #horizontal flip applied to both rbg and depth images with 50% probability
        self.transform_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(width=544, height=416, always_apply=True),
        ], additional_targets={'depth': 'image'}) #additional_targets allows us to apply the same transformation to depth image

        #random transformation that are applied to rbg image only 
        self.transform_rgb_only = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5), #nrightnes scaled to  [.75 - 1.25] with .5 probability
            A.RandomGamma()

        ])
        #normalize values from 0-1 
        #Alburmentation uses the following to normalize img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        self.normalize_tr = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)
        ])

        self.random_crop = A.Compose([
            A.RandomCrop(width=544, height=416, always_apply=True)
        ], additional_targets={'depth': 'image'}) #additional_targets allows us to apply the same transformation to depth image)
            

    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
         #index list and split by space
        rgb_path, depth_path, focal_length = self.file_paths[idx].split()
        

        #We need to provide the full path of the images when working in CAML, we prepend the root path
        # to each image path 
        if self.isCAML:
            if self.mode == 'train':
                #path =  '/scratch365/palarcon/Scene-Reconstruction/BTS/dataset/train/sync/'
                path = 'BTS/dataset/train/sync/'
                rgb_path = rgb_path[1:] #train_text.txt contains a "/" at the beginning of each path, so we need to remove it
                depth_path = depth_path[1:]
            else: #we are in test mode
                #path = '/scratch365/palarcon/Scene-Reconstruction/BTS/'
                path = 'BTS/dataset/train/sync'            
        
        else: #we are working on our local machine
            if self.mode == 'train':#we are working in local machine
                path = 'BTS/dataset/train/sync/'
                rgb_path = rgb_path[1:] #train_text.txt contains a "/" at the beginning of each path, so we need to remove it
                depth_path = depth_path[1:]
            else: 
                path = ''


        rgb_path = os.path.join(path, rgb_path)
        depth_path = os.path.join(path, depth_path)

       

        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)

        # To avoid blank boundaries due to pixel registration
        depth = depth.crop((43, 45, 608, 472))
        rgb = rgb.crop((43, 45, 608, 472))

        #We do not rotate or augment when doing inference
        if self.mode == 'train':
            #performs random rotation within the range of [-2.5, 2.5] degrees
            angle = np.random.uniform(-2.5, 2.5)
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            depth = depth.rotate(angle, resample=Image.NEAREST)



        #convert to numpy array
        rgb = np.asarray(rgb, dtype=np.float32) / 255.0 #normalize values from 0-1  https://medium.com/analytics-vidhya/a-tip-a-day-python-tip-8-why-should-we-normalize-image-pixel-values-or-divide-by-255-4608ac5cd26a
        #depth shape is (480, 640) needs a channel dimension, converts to (480, 640, 1)
        depth = np.asarray(depth, dtype=np.float32) 
        depth = np.expand_dims(depth, axis=2) / 1000 #add channel dimension & downscale to original values

        if self.mode == 'train':
            #apply augmentations: horizontal flip, random crop, random brightness, random gamma, and random rgb shift
            aug_rgb, aug_depth = self.augmentation(rgb, depth)
            #normalize values from 0-1
            aug_rgb = self.normalize_tr(image=aug_rgb)['image']
        elif self.mode == 'test':
            #normalize values from 0-1
            data = self.random_crop(image=rgb, depth=depth)
            aug_depth = data['depth']
            aug_rgb = self.normalize_tr(image=data['image'])['image']
            

        #convert numpy array to tensor
        aug_rgb = np.transpose(aug_rgb, (2, 0, 1)) #convert from (H, W, C) to (3, H, W)
        aug_depth = np.transpose(aug_depth, (2, 0, 1))
        rgb_tensor = torch.from_numpy(aug_rgb)
        depth_tensor = torch.from_numpy(aug_depth)
        focal_tensor = torch.tensor(self.focal_length)

        result = {'image': rgb_tensor, 'depth':depth_tensor, 'focal': focal_tensor}
       

        return result


    def augmentation(self, rgb, depth):

        #apply horizontal flip to both rgb and depth images
        data = self.transform_train(image=rgb, depth=depth)
   

        #apply random brightness, gamma, and rgb shift to rgb image only
        data['image'] = self.transform_rgb_only(image=data['image'])['image']

        #augment RBG colors
        image = data['image']
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        data['image'] *= color_image
        data['image'] = np.clip(data['image'], 0, 1)

        return data['image'], data['depth']



def viz_item(item):
    image = item['image']
    depth = item['depth']
    #original = item['original']

   
    depth = np.transpose(np.array(depth), (1, 2, 0)) #convert from (1, H, W) to (H, W, 1)

    image = unnormalize(image)*255 #convert from normalized to original values
    #image = inverse_normalize(image=np.array(image))['image']
    image = np.transpose(np.array(image), (1, 2, 0)) #convert from (3, H, W) to (H, W, 3)
    

    display = [ image, depth]
    titles = ['RGB', 'Depth']

    fig = plt.figure(figsize=(15, 15))
    for i in range(len(display)):
        plt.subplot(1, 2, i+1)
        plt.imshow(display[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.show()


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)



if __name__ == "__main__":
    nyu_v2 = NYUDatasetRGBD(TRAIN_PATHS, mode='train', isCAML=False)
    print("Dataset Length: ", len(nyu_v2))

    image_index = 2
    viz_item(nyu_v2.__getitem__(image_index))

    depth = nyu_v2.__getitem__(image_index)['depth']
    img = nyu_v2.__getitem__(image_index)['image']

    print("Image Shape" , img.shape)
    print("Depth Shape: ", depth.shape)
    print("Depth Type: ", depth.dtype)


    print("Depth min: ", depth.min())
    print("Depth max: ", depth.max())

    depth = torch.where(depth < 1e-3, depth * 0 + 1e3, depth)
    normalized = normalize_result(1/depth)
    print("Normalized min: ", normalized.min())
    print("Normalized max: ", normalized.max())

    normalized_depth = np.transpose(np.array(normalized), (1, 2, 0))
    print("Normalized Depth Shape: ", normalized_depth.shape)
    plt.imshow(normalized_depth, cmap='jet')
    plt.show()



    

