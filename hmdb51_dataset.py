import torch
import torchvision.transforms as transforms

import os
from os.path import join
import glob
import cv2
import random
import numpy as np

from PIL import Image
import torch.utils.data as data

class HMDB51(data.Dataset):
    def __init__(self, train=True, transform=None, ratio=0.7, Spatial=True):
        self.transform = transform
        self.train = train
        self.data = {}
        self.label_index = {}
        self.ratio = ratio
        video_list = []
        video_folder = '../hmdb51_org'
        data_folder = '../data' 
        path_list = [join(data_folder,'train'), join(data_folder,'validation'), join(data_folder,'test')]
        self.labels = sorted(os.listdir(join(video_folder)))

        # label indexing, {'brush_hair': array(0}, ...}
        self.label_index = {label : np.array(i) for i, label in enumerate(self.labels)}
        # (video -> image)
        if not os.path.exists(join(data_folder,'train')):
            for label in self.labels:
                video_list.append([avi for avi in glob.iglob(join(video_folder,label,'*.avi'), recursive=True)])
                for path in path_list:
                    os.makedirs(join(path,'spatial',label), exist_ok=True)
                    os.makedirs(join(path,'temporal',label), exist_ok=True)
            # len(video_list) = 51, len(videos) = how many videos in each label
            for videos in video_list:
                train_num = round(len(videos)*(self.ratio**2))
                test_num = round(len(videos)*(1-self.ratio))
                for i, video in enumerate(videos):
                    if i < train_num:
                        self.video2frame(video, join(path_list[0],'spatial'), join(path_list[0],'temporal'))
                    elif train_num <= i < (len(videos)-test_num):
                        self.video2frame(video, join(path_list[1],'spatial'), join(path_list[1],'temporal'))
                    else:
                        self.video2frame(video, join(path_list[2],'spatial'), join(path_list[2],'temporal'))
        # {image: label}
        if train:
            mode = 'train'
        else:
            mode = 'test'
        if Spatial:
            f_name = 'spatial'
        else:
            f_name = 'temporal'
        image_list = glob.glob(join(data_folder, mode, f_name,'**','*.jpg'), recursive=True)
        for image in image_list:
            self.data[image] = self.label_index[image.split('/')[-2]]
 
        # train, test split
        split_idx = int(len(image_list) * ratio)
        random.shuffle(image_list)
        self.train_image, self.test_image = image_list[:split_idx], image_list[split_idx:]

        self.train_label = [self.data[image] for image in self.train_image]
        self.test_label = [self.data[image] for image in self.test_image] 

    def video2frame(self, video, spatial_path, temporal_path, count=0):
        folder_name, video_name= video.split('/')[-2], video.split('/')[-1]

        capture = cv2.VideoCapture(video)
        #get_frame_rate = round(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 16)

        _, frame = capture.read() # 다음 프레임과 옵티컬 플로우를 구하기 위해 첫 프레임 저장
        prvs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        hsv = np.zeros_like(frame) # Farneback 알고리즘 이용하기 위한 초기화
        hsv[..., 1] = 255 # 초록색 바탕 설정

        while True:
            ret, image = capture.read()
            if not ret:
                break

            count += 1
            #if(int(capture.get(1)) % get_frame_rate == 0):
            fname = '/{0}_{1:05d}_S.jpg'.format(video_name, count)
            cv2.imwrite(join(spatial_path,folder_name,fname), image)

            #if(int(capture.get(1)) % get_frame_rate == 0):
            next_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            fname = '/{0}_{1:05d}_T.jpg'.format(video_name, count)
            cv2.imwrite(join(temporal_path,folder_name,fname), rgb) 

            prvs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        print('{} spatial images are extracted in {}'.format(count, join(spatial_path,folder_name,video_name)))
        print('{} temporal images are extracted in {}.'.format(count, join(temporal_path,folder_name,video_name)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_image[index], self.train_label[index]

        else:
            img, target = self.test_image[index], self.test_label[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_image)
        else:
            return len(self.test_image)
