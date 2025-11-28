import cv2
import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class Dataset(object):

    def __init__(self, data_dir, fold, input_size=None, normalize_mean=None, normalize_std=None):
        # -------------------load data list,[class,video_name]-------------------
        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]
        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]
        if input_size is None:
            input_size = [400, 400]
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.split = fold
        self.history_mask_list = [None] * self.__len__()

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold in fold_list:

            f = open(os.path.join(self.data_dir, 'Binary_map', 'split%d.txt' % fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("_")
                img_name = item2[0]
                cat = int(item2[1])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(input_size)
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]

        support_img_list = self.binary_pair_list[sample_class]
        while True:
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        support_name_rgb = support_name
        support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
        support_name_th = support_name.replace('.png', '_th.png')
        support_name_d = support_name.replace('.png', '_d.png')
        input_size = self.input_size[0]

        scaled_size = int(random.uniform(1, 1.5) * input_size)

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                             interpolation=InterpolationMode.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                            interpolation=InterpolationMode.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                           interpolation=InterpolationMode.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                          interpolation=InterpolationMode.BILINEAR)
        flip_flag = random.random()

        if self.history_mask_list[index] is None:
            history_mask = torch.zeros(400, 400).fill_(0)
        else:
            history_mask = self.history_mask_list[index]

        image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        image_th = Image.fromarray(cv2.cvtColor(image_th, cv2.COLOR_BGR2RGB))
        image_d = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_d))
        image_d = Image.fromarray(cv2.cvtColor(image_d, cv2.COLOR_BGR2RGB))

        support_th = self.normalize(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_th))))

        support_d = self.normalize(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, image_d))))

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb)).convert('RGB')))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name)))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]


        scaled_size = input_size
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                             interpolation=InterpolationMode.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                            interpolation=InterpolationMode.NEAREST)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                           interpolation=InterpolationMode.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                          interpolation=InterpolationMode.BILINEAR)

        flip_flag = 0
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png', '_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_dq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_d))
        image_dq = Image.fromarray(cv2.cvtColor(image_dq, cv2.COLOR_BGR2RGB))

        query_th = self.normalize(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_thq))))

        query_d = self.normalize(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, image_dq))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag, Image.open(
                        os.path.join(self.data_dir, 'seperated_images', query_name_rgb)).convert('RGB')))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class), query_name)))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return query_rgb, query_th, query_d, query_mask.long(), support_rgb, support_th, support_d, support_mask.long(), \
               sample_class - 1, index

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return len(self.new_exist_class_list)
