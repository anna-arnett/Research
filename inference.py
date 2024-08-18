from bts import BTS
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2 
from dataloader import NYUDatasetRGBD

TEST_DIR = 'BTS/dataset/test_paths.txt'
#MODEL_DIR = 'BTS/models/model-49'
MODEL_DIR = 'BTS/logs/checkpoints/model-49'
TRAIN_DIR = 'BTS/dataset/train_paths.txt'


#converts the depth_gt and depth_est from batch torchs of size [1, 1, H, W] to numpy arrays of size [H, W, 1]
def tensor_to_numpy(depth_gt, depth_est):

    #remove batch dimension using squeeze, detach from cuda, transfer to cpu and convert to numpy
    depth_est = depth_est.squeeze(1).detach().cpu().numpy()
    #convert from (1, H, W) to (H, W, 1)
    depth_est = np.transpose(np.array(depth_est), (1, 2, 0)) 

    #process ground_truth
    depth_gt = depth_gt.squeeze(1).numpy()
    #convert from (1, H, W) to (H, W, 1)
    depth_gt = np.transpose(np.array(depth_gt), (1, 2, 0)) 

    #print(f'depth_gt shape: {depth_gt.shape}')
    #print(f'depth_est shape: {depth_est.shape}')

    return depth_gt, depth_est

#takes as"' input an array of depth_estimate and depth_ground_truth and saved them to
#an inference folder
def save_images(depth_estimate):
    #first, get the rgb image name in order as they appear in the test_paths.txt file
    image_names = []
    with open(TEST_DIR, 'r') as f:
        rgb_path, _, _ = f.readline().split()
        img_name = rgb_path.split('/')[-1] #get the image name
        image_names.append(img_name)

    #create an inference folder to save the images (if one already does not exist)
    est_dir = 'BTS/depth_estimates'
    if not os.path.isfile(est_dir):
        os.makedirs(est_dir, exist_ok=True)

    #save the images
    for index in range(len(depth_estimate)):
        
        est_filename = est_dir + '/est_' + image_names[index].split('_')[-1] + '.png'
        est = depth_estimate[index] * 1000.0 #upscaling the depth values 
        est_uint16 = est.astype(np.uint16)

        cv2.imwrite(est_filename, est_uint16)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def eval(gt_depth, est_depth):

    est_depth[ est_depth < 1e-3] = 1e-3
    est_depth[ est_depth > 10] = 10
    est_depth[np.isinf( est_depth )] = 10
    est_depth[np.isnan( est_depth )] = 1e-3

    valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth  < 10)
    eval_mask = np.zeros(valid_mask.shape)
    eval_mask[45:471, 41:601] = 1
    valid_mask = np.logical_and(valid_mask, eval_mask)
    measures = compute_errors(est_depth[valid_mask], gt_depth [valid_mask])

    return measures

if __name__ == "__main__":

    
    dataset = NYUDatasetRGBD(TEST_DIR, mode='test', isCAML=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    to_device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = BTS(max_depth=10, bts_size=512)
    checkpoint = torch.load(MODEL_DIR, map_location=to_device)
    model.load_state_dict(checkpoint['model'], strict=False )
    #model.eval() #using model.eval gives crazy wrong results, ask Prof. likely due to batchnorm or batchsize 
    model.to(to_device)

    depth_estimates = []
    depth_ground_truth = []

    print(f'Predicting depth for {len(dataset)} images...')


    metric_measures = [0,0,0,0,0,0,0,0,0] #[silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]
    inference_time = 0
    with torch.no_grad():
        for step, sampled_batch in enumerate(dataloader):
            print(f'Evaluation Image No. {step}')
            image = sampled_batch['image'].to(to_device)
            depth_gt = sampled_batch['depth'].to(to_device)      
            focal = sampled_batch['focal'].to(to_device)

            start = time.time()
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            end = time.time()
            inference_time += (end-start)

            depth_gt = depth_gt.cpu().numpy().squeeze()
            depth_est = depth_est.cpu().numpy().squeeze()
            measures = eval(depth_gt,depth_est)
            metric_measures = np.add(measures, metric_measures)


    metric_name = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
    for i in range(len(metric_name)):
        print(f'{metric_name[i]}: {metric_measures[i]/len(dataset)}')
    
    print(f'Finished predicting depth for {len(dataset)} images in {inference_time} seconds')
    #save estimate images to file
    #print("Saving Depth Estimates to file...")
    #save_images(depth_estimates)

    #plt.imshow(depth_est)

