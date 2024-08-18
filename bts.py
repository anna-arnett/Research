import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import albumentations as A
from torch.utils.tensorboard import SummaryWriter
#Importing Custom scripts
import sys
#sys.path.append('/scratch365/palarcon/Scene-Reconstruction/BTS/')
sys.path.append('./BTS/')
from encoder import Encoder
from decoder import Decoder, silog_loss, weights_init_xavier
from dataloader import NYUDatasetRGBD

#uncomment when training in CRC
#TRAIN_DIR = '/scratch365/palarcon/Scene-Reconstruction/BTS/dataset/train_paths.txt'
#TEST_DIR = '/scratch365/palarcon/Scene-Reconstruction/BTS/dataset/test_paths.txt'
#LOG_DIR = '/scratch365/palarcon/Scene-Reconstruction/BTS/logs/'
#TENSORBOARD_DIR = '/scratch365/palarcon/Scene-Reconstruction/BTS/logs/tensorboard/'
#MODEL_DIR = '/scratch365/palarcon/Scene-Reconstruction/BTS/logs/checkpoints/'
#uncommment when training locally

TRAIN_DIR = 'BTS/dataset/train_paths.txt'
TEST_DIR = 'BTS/dataset/test_paths.txt'
LOG_DIR = 'BTS/logs/'
TENSORBOARD_DIR = 'BTS/logs/tensorboard/'
MODEL_DIR = 'BTS/logs/checkpoints/'


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class BTS(nn.Module):

    def __init__(self, max_depth, bts_size):
        super(BTS, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(max_depth, self.encoder.feat_out_channels, bts_size)


    def forward(self, input_image, focal):
        skip_features = self.encoder(input_image)
        depth_estimates = self.decoder(skip_features, focal)

        return depth_estimates



class Controller():
    def __init__(self, num_epochs, learning_rate, batch_size, max_depth, bts_size, tensorboard_dir, model_dir, train_dir, save_freq, tensorboard_freq):

        self.train_dataset = NYUDatasetRGBD(TRAIN_DIR, mode='train', isCAML=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.test_dataset = NYUDatasetRGBD(TEST_DIR, mode='test', isCAML=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        self.model_dir = model_dir
        self.save_freq = save_freq
        self.tensorboard_freq = tensorboard_freq


        self.num_epochs = num_epochs 
        self.global_step = 0
        self.steps_per_epoch = len(self.train_dataset) 
        self.num_total_steps = self.steps_per_epoch * self.num_epochs
        self.epoch = self.global_step // self.steps_per_epoch
        self.end_learning_rate = .01 * learning_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.writer = SummaryWriter(tensorboard_dir, flush_secs=30)
        self.num_log_images = batch_size
        self.eval_global_step = 0

        #check if GPU is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.criterion = silog_loss(variance_focus=0.85)

        self.model = BTS(max_depth, bts_size).to(self.device)
        self.model.decoder.apply(weights_init_xavier)
        self.optimizer = torch.optim.AdamW([{'params': self.model.encoder.parameters(), 'weight_decay': 1e-2 },
                                            {'params': self.model.decoder.parameters(), 'weight_decay': 0.0}],
                                             lr=1e-4, eps= 1e-3)

        self.inv_normalize = A.Compose([
            A.Normalize(    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
        ])

        self.start_time = time.time()
        self.duration = 0
 


    def normalize_result(self, value, vmin=None, vmax=None):
        value = value.cpu().numpy()[0, :, :]

        vmin = value.min() if vmin is None else vmin
        vmax = value.max() if vmax is None else vmax

        if vmin != vmax:
            value = (value - vmin) / (vmax - vmin)
        else:
            value = value * 0.

        return np.expand_dims(value, 0)

    def train(self):
        
        self.model.decoder.apply(weights_init_xavier)
        while self.epoch < self.num_epochs:
            #training loop
            self.model.train()
            for step, sample_batched in enumerate(self.train_dataloader):
                self.optimizer.zero_grad() #zero out gradients
                before_op_time = time.time()

                
                image = torch.autograd.Variable(sample_batched['image'].to(self.device))
                focal = torch.autograd.Variable(sample_batched['focal'].to(self.device))
                depth_gt = torch.autograd.Variable(sample_batched['depth'].to(self.device))

                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.model(image, focal)

                mask = depth_gt > .1

                loss = self.criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
                loss.backward()

                for param_group in self.optimizer.param_groups:
                    current_lr = (self.learning_rate - self.end_learning_rate) * (1 - self.global_step / self.num_total_steps) ** 0.9 + self.end_learning_rate
                    param_group['lr'] = current_lr

                self.optimizer.step()

                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1
                print(f'[epoch][s/s_per_e/gs]: [{self.epoch}][{step}/{self.steps_per_epoch}/{self.global_step}], lr: {current_lr}, loss: {loss}')


                self.duration += time.time() - before_op_time

                if self.global_step % self.tensorboard_freq == 0 and self.global_step:

                    self.to_tensorboard(loss, current_lr,  depth_gt, depth_est, reduc1x1, lpg2x2, lpg4x4, lpg8x8, image, mode='train')
                    self.to_console(self.tensorboard_freq, loss)

                #saves checkpoint every  epoch, for a total of 150 checkpoints
                self.global_step += 1

            #evaluation loop every epoch
            self.model.eval()
            metric_measures = [0,0,0,0,0,0,0,0,0] #[silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]
            for step, sample_batched in enumerate(self.test_dataloader):
                print("step: ", step)
                with torch.no_grad():
                    image = torch.autograd.Variable(sample_batched['image'].to(self.device))
                    focal = torch.autograd.Variable(sample_batched['focal'].to(self.device))
                    depth_gt = torch.autograd.Variable(sample_batched['depth'].to(self.device))

                    _, _, _, _, depth_est = self.model(image, focal)


                    if step % self.tensorboard_freq == 0 and step:

                        self.to_tensorboard(None, None,  depth_gt, depth_est, None, None, None, None, image, mode='test')

                    depth_gt = depth_gt.cpu().numpy().squeeze()
                    depth_est = depth_est.cpu().numpy().squeeze()

                depth_est[ depth_est < 1e-3] = 1e-3
                depth_est[ depth_est > 10] = 10
                depth_est[np.isinf( depth_est)] = 10
                depth_est[np.isnan( depth_est)] = 1e-3

                valid_mask = np.logical_and(depth_gt > 1e-3, depth_gt < 10)
                eval_mask = np.zeros(valid_mask.shape)
                eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
                measures = compute_errors(depth_est[valid_mask], depth_gt[valid_mask])

                metric_measures = np.add(measures, metric_measures)

                        
                self.eval_global_step += 1
                
            epoch_metrics = metric_measures/len(self.test_dataloader)

            metric = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
            for i in range(len(metric)):
                self.writer.add_scalar(f'eval/{metric[i]}', epoch_metrics[i], self.epoch)

            print('Saving checkpoint...')
            self.save_checkpoint(self.model_dir)
            self.epoch += 1

        self.writer.close()

    def save_checkpoint(self, log_directory):
        checkpoint = {'global_step': self.global_step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, log_directory + f'model-{self.epoch}')

    def to_tensorboard(self, loss, current_lr, depth_gt, depth_est, reduc1x1, lpg2x2, lpg4x4, lpg8x8, image, mode='train'):

        print('Saving to tensorboard')
        if mode == 'train':
            self.writer.add_scalar('silog_loss', loss, self.global_step)
            self.writer.add_scalar('learning_rate', current_lr, self.global_step)
            #self.writer.add_scalar('var average', var_sum.item()/var_cnt, self.global_step)
            depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
            for i in range(self.num_log_images):
                self.writer.add_image(f'train/depth_gt_{i}', self.normalize_result(1/depth_gt[i, :, :, :].data), self.global_step)
                self.writer.add_image(f'train/depth_est_{i}', self.normalize_result(1/depth_est[i, :, :, :].data), self.global_step)
                self.writer.add_image(f'train/reduc1x1_{i}', self.normalize_result(1/reduc1x1[i, :, :, :].data), self.global_step)
                self.writer.add_image(f'train/lpg2x2_{i}', self.normalize_result(1/lpg2x2[i, :, :, :].data), self.global_step)
                self.writer.add_image(f'train/lpg4x4_{i}', self.normalize_result(1/lpg4x4[i, :, :, :].data), self.global_step)
                self.writer.add_image(f'train/lpg8x8_{i}', self.normalize_result(1/lpg8x8[i, :, :, :].data), self.global_step)
                self.writer.add_image(f'train/image_{i}', (unnormalize(tensor=image[i, :, :, :])*255).data, self.global_step)
        else:
            depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
            self.writer.add_image(f'eval/depth_gt_', self.normalize_result(1/depth_gt[0, :, :, :].data), self.eval_global_step)
            self.writer.add_image(f'eval/depth_est_', self.normalize_result(1/depth_est[0, :, :, :].data), self.eval_global_step)
            self.writer.add_image(f'eval/image_', (unnormalize(tensor=image[0, :, :, :])*255).data, self.eval_global_step)


        self.writer.flush()



    def to_console(self, log_frequency, loss):
        examples_per_sec = self.batch_size / self.duration * log_frequency
        self.duration = 0
        time_sofar = (time.time() - self.start_time) / 3600
        training_time_left = (self.num_total_steps / self.global_step - 1.0) * time_sofar
        print_string = ' examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
        print(print_string.format(examples_per_sec, loss, time_sofar, training_time_left))


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
    

if __name__ == '__main__':


    '''  
     Let's check our log directories exists, otherwise create it.
     Here is how we will structure our log directories:

     BTS
     --logs
     -----checkpoints
     ---------checkpoint_1.pth
     ---------checkpoint_N.pth
     -----tensorboard
     ---------events.out.tfevents.1600000000.localhost
     -----condor
     ---------condor.out
     ---------condor.err
    '''
    
  
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(TENSORBOARD_DIR):
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    

    controller_args = {
        'num_epochs': 1,
        'learning_rate': 1e-4,
        'batch_size': 4,
        'max_depth': 10,
        'bts_size': 512, 
        'tensorboard_dir': TENSORBOARD_DIR,
        'model_dir': MODEL_DIR,
        'train_dir': TRAIN_DIR,
        'save_freq': 1,
        'tensorboard_freq': 10,
    }




    bts_controller = Controller(**controller_args)
    bts_controller.train()





