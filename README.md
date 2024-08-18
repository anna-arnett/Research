Written by Pedro Alarcon Granadeno: 

For any issues, feel free to contant me: palarcon@nd.edu

# Deliverable #4 & #5

Instructions on how to execute an evaluation sample. 

1. Clone this repo and install dependencies: pytorch, tensorflow, numpy, cv2, matplotlib, alburmentations, tqdm, pillow
2. Downlod the [trained model](https://drive.google.com/file/d/1v3nGIww4SE79sLOCdvt-lom95-4Qgss8/view?usp=sharing) and place it inside BTS/logs/checkpoints/
3. Download the [test set](https://drive.google.com/drive/folders/11nymeVH1XWtJow6hI7khmxH3QqDKDqwm?usp=share_link) and place it inside BTS/dataset/
4. Go to and run eval_sample.ipynb. You can change the index to make prediction on any of the test images (0-653)



# Deliverable #3 

## Executing Augmentations for Depth Estimation. 

I leveraged Pytorch's Dataloaders and the Alburmentations module to create an augmentation and preprocessing pipeline for my training images. In order to test the working code, please execute the following instructions. Note that you will need following modules: ```Pytorch, Alburmentations, PIL, Matplotlib and Numpy```. 

1. Download the BTS folder from this [Google Drive link](https://drive.google.com/drive/folders/1-BjOowa_i_k6Xr3ezjAJk4PArYFLTjlu?usp=sharing). The folder should contain a dataset folder with train and test images, a dataloader.py script and train_paths.txt file.
2. Open the dataloader.py using your favorite IDE.
3. Execute the dataloader.py code. You can change the image being augmented by providing a different index in line 114. 


## Executing the DPT Baseline Scripts

1. Clone the [DPT Repository](https://github.com/isl-org/DPT)
2. Install DPT dependencies as described in their README.md
3. Download the [NYU-fined tuned DPT model](https://drive.google.com/file/d/1iJSJbf0FezhYKhEKXvjaGvf1v-Dz3OcT/view?usp=sharing) and place it inside the "weights" folder.
4. Replace the "input" folder with our training data found [here](https://drive.google.com/drive/folders/1OVm3Jfd4wqvazSOPmf8cHj7hw8bUuYwf?usp=sharing) This folder will also contain the ground truth depth images inside the ```gt``` folder.
5. Now, Please choose **one** of the two following options. Both options produce the exact same baseline result. 
    * You may reproduce the depth estimation results outputted by the DPT model by executing ```python run_monodepth.py --model_type dpt_hybrid_nyu --absolute_depth``` inside the root folder. GPU is recommended but not necessary. Inference without GPU should take about 45 minutes. 
    * Or you may simply download the [results we obtained](https://drive.google.com/drive/folders/1kBcQjvfQ7t_JoOi-9ZSXuJtvURbQTmct?usp=sharing). Please replace ```output_monodepth``` folder with our results. 

6. Download the [DPT_Baseline.ipynb](https://github.com/TonyAlarcon/Computer-Vision-60535/blob/db90dec0ee4cfb5cea0efe97c505a130d90fc961/BTS/DPT_Baseline.ipynb) and place it inside the project's root folder and simply execute. It should print the inferenced results detailed in our report. 


