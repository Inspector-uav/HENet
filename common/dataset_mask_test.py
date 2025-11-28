import random
import os
import torchvision
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F


class Dataset(object):

    def __init__(self, data_dir, fold, input_size=[400, 400], normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225], shot=1):

        self.data_dir = data_dir

        self.input_size = input_size
        self.shot = shot
        self.history_mask_list = [None] * self.__len__()

        self.chosen_data_list = self.get_new_exist_class_dict(fold=fold)

        self.split = fold
        self.binary_pair_list = self.get_binary_pair_list()
        self.query_class_support_list = [None] * len(self.chosen_data_list)

        for index in range(len(self.chosen_data_list)):
            query_name = self.chosen_data_list[index][0]
            sample_class = self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]
            support_names = []
            while True:
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]

                if query_name != support_name and support_name not in support_names: support_names.append(support_name)
                if len(support_names) == self.shot: break
            self.query_class_support_list[index] = [query_name, sample_class, support_names]

        if self.split == 3:
            self.sub_val_list = list(range(13, 17))
        elif self.split == 2:
            self.sub_val_list = list(range(9, 13))
        elif self.split == 1:
            self.sub_val_list = list(range(5, 9))
        elif self.split == 0:
            self.sub_val_list = list(range(1, 2))  # [1]
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % (fold)))
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
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):
        binary_pair_list = {}
        for Class in range(1, 2):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
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
        support_image_list = []
        support_label_list = []
        support_image_th_list = []
        support_image_d_list = []

        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]
        support_name1 = self.query_class_support_list[index][2]

        for k in range(self.shot):

            support_name = support_name1[k]

            input_size = self.input_size[0]

            scaled_size = int(random.uniform(1, 1.5) * input_size)
            scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                                 interpolation=Image.NEAREST)
            scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                                interpolation=Image.BILINEAR)
            scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                               interpolation=Image.BILINEAR)
            scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                               interpolation=Image.BILINEAR)

            flip_flag = random.random()
            support_name_rgb = support_name
            support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
            support_name_th = support_name.replace('.png', '_th.png')
            support_name_d = support_name.replace('.png', '_d.png')

            image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
            image_th = Image.fromarray(cv2.cvtColor(image_th, cv2.COLOR_BGR2RGB))
            image_d = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_d))
            image_d = Image.fromarray(cv2.cvtColor(image_d, cv2.COLOR_BGR2RGB))

            support_th = self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_th)))

            support_th = self.normalize(support_th)

            support_d = self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, image_th)))

            support_d = self.normalize(support_d)

            support_rgb = self.normalize(self.ToTensor(
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
            support_image_list.append(support_rgb)
            support_label_list.append(support_mask)
            support_image_th_list.append(support_th)
            support_image_d_list.append(support_d)

        scaled_size = 400

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()

        if self.history_mask_list[index] is None:

            history_mask = torch.zeros(400, 400).fill_(0.0)  # 创建一个三维数组,用 0.0 填充

        else:

            history_mask = self.history_mask_list[index]

        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png', '_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_dq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_d))
        image_dq = Image.fromarray(cv2.cvtColor(image_dq, cv2.COLOR_BGR2RGB))

        query_th = self.ToTensor(
            scale_transform_th(
                self.flip(flip_flag, image_thq)))
        query_th = self.normalize(query_th)

        query_d = self.ToTensor(
            scale_transform_d(
                self.flip(flip_flag, image_dq)))
        query_d = self.normalize(query_d)

        query_rgb = self.normalize(self.ToTensor(
            scale_transform_rgb(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'seperated_images', query_name_rgb)).convert('RGB')))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class), query_name)))))

        margin_h = random.randint(0, scaled_size - input_size)  # 0
        margin_w = random.randint(0, scaled_size - input_size)  # 0

        query_rgb = query_rgb[:, margin_h:margin_h + input_size,
                    margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        s_xs = support_image_list
        s_ys = support_label_list
        s_ths = support_image_th_list
        s_ds = support_image_d_list

        s_x = s_xs[0].unsqueeze(0)  # torch.Size([1, 3, 473, 473])
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)

        s_y = s_ys[0]
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i], s_y], 0)

        s_th = s_ths[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_th = torch.cat([s_ths[i].unsqueeze(0), s_th], 0)

        return query_rgb, query_th, query_d, query_mask.long(), support_rgb, support_th, support_d, support_mask.long(), \
               sample_class - 1, index, \
               os.path.join(self.data_dir, 'seperated_images', support_name_rgb), \
               os.path.join(self.data_dir, 'seperated_images', support_name_th), \
               os.path.join(self.data_dir, 'seperated_images', query_name_rgb), \
               os.path.join(self.data_dir, 'seperated_images', query_name_th)

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return 20
