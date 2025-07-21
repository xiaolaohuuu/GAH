import torch
import numpy as np

import os
import scipy.io as scio
from collections import namedtuple
from scipy.io import loadmat
import torch.utils.data as data
import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, images, tags, labels, test=None):
        self.test = test
        all_index = np.arange(tags.shape[0])
        if opt.flag == 'mir':
            query_index = all_index[opt.db_size:]
            training_index = all_index[:opt.training_size]
            db_index = all_index[:opt.db_size]
        else:
            query_index = all_index[:opt.query_size]
            training_index = all_index[opt.query_size: opt.query_size + opt.training_size]
            db_index = all_index[opt.query_size:]

        if test is None:
            train_images = images[training_index]
            train_tags = tags[training_index]
            train_labels = labels[training_index]
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[query_index]
            self.db_labels = labels[db_index]
            if test == 'image.query':
                self.images = images[query_index]
            elif test == 'image.db':
                self.images = images[db_index]
            elif test == 'text.query':
                self.tags = tags[query_index]
            elif test == 'text.db':
                self.tags = tags[db_index]

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                torch.from_numpy(self.images[index].astype('float32')),
                torch.from_numpy(self.tags[index].astype('float32')),
                torch.from_numpy(self.labels[index].astype('float32'))
            )
        elif self.test.startswith('image'):
            return torch.from_numpy(self.images[index].astype('float32'))
        elif self.test.startswith('text'):
            return torch.from_numpy(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return torch.from_numpy(self.labels.astype('float32'))
        else:
            return (
                torch.from_numpy(self.query_labels.astype('float32')),
                torch.from_numpy(self.db_labels.astype('float32'))
            )

paths_COCO = {
    # MSCOCO
    #TODO:K=1
    # 'text_file': "/data/COCO-2017/k_1/train_text_1shot.mat",
    # 'train_label_file': "/data/COCO-2017/k_1/train_label_1shot_repeat10.mat",
    # 'images_file': "/data/COCO-2017/k_1/train_image_1shot.mat",
    
    # # 'textaug_file': "/data/COCO-2017/k_1/text_onlytext_label_1shot.mat",
    # # 'textaug_file': "/data/COCO-2017/k_1/augnewtext_1_top20.mat",
    # 'textaug_file': "/data/COCO-2017/k_1/augnewtext_1_last20.mat",

    # # 'imgaug_file': "/data/COCO-2017/k_1/image_onlytext_label_1shot.mat",
    # # 'imgaug_file': "/data/COCO-2017/k_1/augnew_1_top20.mat",
    # 'imgaug_file': "/data/COCO-2017/k_1/augnew_1_last20.mat",
    #TODO:K=2
    # 'text_file': "/data/COCO-2017/train_text_2shot.mat",
    # 'train_label_file': "/data/COCO-2017/train_label_2shot_repeat10.mat",
    # 'images_file': "/data/COCO-2017/train_image_2shot.mat",

    # # 'textaug_file': "/data/COCO-2017/debug_image/text_onlytext_label_2shot.mat",
    # # 'textaug_file': "/data/COCO-2017/k_2/augnewtext_2_last20.mat",
    # 'textaug_file': "/data/COCO-2017/k_2/augnewtext_2_top20.mat",
    # # 'textaug_file': "/data/COCO-2017/k_2/augtext_2_last20.mat",

    # # 'imgaug_file': "/data/COCO-2017/debug_image/image_onlytext_label_2shot.mat",
    # # 'imgaug_file': "/data/COCO-2017/k_2/augnew_2_last20.mat",
    # 'imgaug_file': "/data/COCO-2017/k_2/augnew_2_top20.mat",
    # # 'imgaug_file': "/data/COCO-2017/k_2/aug_2_last20.mat",
    #TODO:K=4
    # 'text_file': "/data/COCO-2017/train_text_4shot.mat",
    # 'train_label_file': "/data/COCO-2017/train_label_4shot_repeat10.mat",
    # 'images_file': "/data/COCO-2017/train_image_4shot.mat",

    # # 'textaug_file': "/data/COCO-2017/k_4/aug_text_4shot.mat", 
    # # 'textaug_file': "/data/COCO-2017/k_4/augnewtext_4_top20.mat", 
    # 'textaug_file': "/data/COCO-2017/k_4/augnewtext_4_last20.mat", 

    # # 'imgaug_file': "/data/COCO-2017/k_4/augimage_4shot.mat",
    # # 'imgaug_file': "/data/COCO-2017/k_4/augnew_4_top20.mat",
    # 'imgaug_file': "/data/COCO-2017/k_4/augnew_4_last20.mat",
    #TODO:K=8
    # 'text_file': "/data/COCO-2017/train_text_8shot.mat",
    # 'train_label_file': "/data/COCO-2017/train_label_8shot_repeat10.mat",
    # 'images_file': "/data/COCO-2017/train_image_8shot.mat",

    # # 'textaug_file': "/data/COCO-2017/k_8/aug_text_8shot.mat",
    # 'textaug_file': "/data/COCO-2017/k_8/augnewtext_8_top20.mat",
    # # 'textaug_file': "/data/COCO-2017/k_8/augnewtext_8_last20.mat",
    
    # # 'imgaug_file': "/data/COCO-2017/k_8/augimage_8shot.mat",
    # 'imgaug_file': "/data/COCO-2017/k_8/augnew_8_top20.mat",
    # # 'imgaug_file': "/data/COCO-2017/k_8/augnew_8_last20.mat",
    #TODO:K=16
    # 'text_file': "/data/COCO-2017/train_text_16shot.mat",
    # 'images_file': "/data/COCO-2017/train_image_16shot.mat",
    # 'train_label_file': "/data/COCO-2017/train_label_16shot_repeat10.mat",

    # # 'textaug_file': "/data/COCO-2017/k_16/aug_text_16shot.mat",
    # # 'textaug_file': "/data/COCO-2017/k_16/augnewtext_16_top20.mat",
    # 'textaug_file': "/data/COCO-2017/k_16/augnewtext_16_last20.mat",
    
    # # 'imgaug_file': "/data/COCO-2017/k_16/augimage_16shot.mat",
    # # 'imgaug_file': "/data/COCO-2017/k_16/augnew_16_top20.mat",
    # 'imgaug_file': "/data/COCO-2017/k_16/augnew_16_last20.mat",

    # 'image_testfile': '/data/COCO-2017/query_image.mat',
    # 'text_testfile': '/data/COCO-2017/query_text.mat',
    # 'test_labelfile': '/data/COCO-2017/query_label_index.mat',
    # 'text_retrievalfile': '/data/COCO-2017/retrieval_text.mat',
    # 'images_retrievalfile': '/data/COCO-2017/retrieval_image.mat',
    # 'retrieval_labelfile': '/data/COCO-2017/retrieval_label_index.mat'

    # mir
    #TODO:K=1
    # 'images_file': "/data/Mirflickr_25k/k_1/train_1shot_images.mat", 
    # 'text_file': "/data/Mirflickr_25k/k_1/train_1shot_texts.mat",
    # 'train_label_file': "/data/Mirflickr_25k/k_1/train_1shot_labels_repeat10.mat",
    
    # # 'textaug_file': "/data/Mirflickr_25k/k_1/aug_text_1shot.mat",
    # # 'textaug_file': "/data/Mirflickr_25k/k_1/augtext_1_top20.mat",
    # 'textaug_file': "/data/Mirflickr_25k/k_1/augtext_1_last20.mat",

    # # 'imgaug_file': "/data/Mirflickr_25k/k_1/augimage_1shot.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_1/aug_1_top20.mat",
    # 'imgaug_file': "/data/Mirflickr_25k/k_1/aug_1_last20.mat",

    #TODO:K=2
    # 'text_file': "/data/Mirflickr_25k/k_2/train_2shot_texts.mat",
    # 'train_label_file': "/data/Mirflickr_25k/k_2/train_2shot_labels_repeat10.mat",
    # 'images_file': "/data/Mirflickr_25k/k_2/train_2shot_images.mat",

    # # 'textaug_file': "/data/Mirflickr_25k/k_2/aug_text_2shot.mat",
    # # 'textaug_file': "/data/Mirflickr_25k/k_2/augtext_2_top20.mat",
    # 'textaug_file': "/data/Mirflickr_25k/k_2/augtext_2_last20.mat",
    
    # # 'imgaug_file': "/data/Mirflickr_25k/k_2/augimage_2shot.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_2/aug_2_top20.mat",
    # 'imgaug_file': "/data/Mirflickr_25k/k_2/aug_2_last20.mat",
    #TODO:K=4
    # 'text_file': "/data/Mirflickr_25k/k_4/train_4shot_texts.mat",
    # 'images_file': "/data/Mirflickr_25k/k_4/train_4shot_images.mat",
    # 'train_label_file': "/data/Mirflickr_25k/k_4/train_4shot_labels_repeat10.mat",

    # # 'textaug_file': "/data/Mirflickr_25k/k_4/aug_text_4shot.mat",
    # # 'textaug_file': "/data/Mirflickr_25k/k_4/augtext_4_last20.mat",
    # 'textaug_file': "/data/Mirflickr_25k/k_4/augtext_4_top20.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_4/augimage_4shot.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_4/aug_4_last20.mat",
    # 'imgaug_file': "/data/Mirflickr_25k/k_4/aug_4_top20.mat",
    #TODO:K=8
    # 'text_file': "/data/Mirflickr_25k/k_8/train_8shot_texts.mat",
    # 'images_file': "/data/Mirflickr_25k/k_8/train_8shot_images.mat",
    # 'train_label_file': "/data/Mirflickr_25k/k_8/train_8shot_labels_repeat10.mat",

    # # 'textaug_file': "/data/Mirflickr_25k/k_8/aug_blip_onlylabel_8shot.mat",
    # # 'textaug_file': "/data/Mirflickr_25k/k_8/aug_text_8shot.mat",
    # 'textaug_file': "/data/Mirflickr_25k/k_8/augtext_8_top20.mat",
    # # 'textaug_file': "/data/Mirflickr_25k/k_8/augtext_8_last20.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_8/augimage_only_label_8shot.mat",
    # 'imgaug_file': "/data/Mirflickr_25k/k_8/aug_8_top20.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_8/aug_8_last20.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_8/augimage_8shot.mat",
    # TODO:K=16
    # 'text_file': "/data/Mirflickr_25k/k_16/train_16shot_texts.mat",
    # 'train_label_file': "/data/Mirflickr_25k/k_16/train_16shot_labels_repeat10.mat",
    # 'images_file': "/data/Mirflickr_25k/k_16/train_16shot_images.mat",

    # # 'textaug_file': "/data/Mirflickr_25k/k_16/aug_text_16shot.mat",
    # # 'textaug_file': "/data/Mirflickr_25k/k_16/augtext_16_top20.mat",
    # 'textaug_file': "/data/Mirflickr_25k/k_16/augtext_16_last20.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_16/augimage_16shot.mat",
    # # 'imgaug_file': "/data/Mirflickr_25k/k_16/aug_16_top20.mat",
    # 'imgaug_file': "/data/Mirflickr_25k/k_16/aug_16_last20.mat",

    # 'image_testfile': '/data/Mirflickr_25k/new_test_image.mat',
    # 'text_testfile': '/data/Mirflickr_25k/new_test_text.mat', 
    # 'test_labelfile': '/data/Mirflickr_25k/new_test_labels.mat',
    # 'text_retrievalfile': '/data/Mirflickr_25k/new_retrieval_text.mat',
    # 'images_retrievalfile': '/data/Mirflickr_25k/new_retrieval_image.mat',
    # 'retrieval_labelfile': '/data/Mirflickr_25k/new_retrieval_labels.mat'

    # NUS 
    #TODO:K=1
    'images_file' : "/data/nus_21class/k_1/train_image.mat",
    'train_label_file' : "/data/nus_21class/k_1/train_labels_repeat10.mat",    
    'text_file' : "/data/nus_21class/k_1/train_text.mat",

    # 'textaug_file': "/data/nus_21class/k_1/Blip_text.mat",
    # 'textaug_file': "/data/nus_21class/k_1/augtext_1_top20.mat",
    'textaug_file': "/data/nus_21class/k_1/augtext_1_last20.mat",

    # 'imgaug_file': "/data/nus_21class/k_1/aug_image.mat",
    # 'imgaug_file': "/data/nus_21class/k_1/aug_1_top20.mat",
    'imgaug_file': "/data/nus_21class/k_1/aug_1_last20.mat",
    #TODO:K=2
    # # 'text_file' : "/data/nus_21class/k_2/train_text.mat",
    # 'text_file' : "/data/nus_21class/new_data/k_2/new_train_text.mat",
    # 'images_file' : "/data/nus_21class/new_data/k_2/new_train_image.mat",
    # # 'images_file' : "/data/nus_21class/k_2/train_image.mat",
    # 'train_label_file' : "/data/nus_21class/new_data/k_2/train_labels_repeat10.mat",
    # # 'train_label_file' : "/data/nus_21class/k_2/train_labels_repeat10.mat",

    # # 'textaug_file': "/data/nus_21class/k_2/Blip_text.mat",
    # # 'textaug_file': "/data/nus_21class/new_data/k_2/augtext_2_top20.mat",
    # 'textaug_file': "/data/nus_21class/new_data/k_2/augtext_2_last20.mat",

    # # 'imgaug_file': "/data/nus_21class/k_2/aug_image.mat",
    # # 'imgaug_file': "/data/nus_21class/new_data/k_2/aug_2_top20.mat",
    # 'imgaug_file': "/data/nus_21class/new_data/k_2/aug_2_last20.mat",
    #TODO:K=4
    # 'text_file' : "/data/nus_21class/new_data/k_4/new_train_text.mat",
    # 'train_label_file' : "/data/nus_21class/new_data/k_4/train_labels_repeat10.mat",
    # 'images_file' : "/data/nus_21class/new_data/k_4/new_train_image.mat",

    # # 'textaug_file': "/data/nus_21class/new_data/k_4/aug_text_feature.mat",
    # 'textaug_file': "/data/nus_21class/new_data/k_4/augtext_4_top20.mat",
    # # 'textaug_file': "/data/nus_21class/new_data/k_4/augtext_4_last20.mat",
    # # 'imgaug_file': "/data/nus_21class/new_data/k_4/aug_image_feature.mat",
    # 'imgaug_file': "/data/nus_21class/new_data/k_4/aug_4_top20.mat",
    # # 'imgaug_file': "/data/nus_21class/new_data/k_4/aug_4_last20.mat",
    #TODO:K=8
    # 'text_file' : "/data/nus_21class/new_data/k_8/new_train_text.mat",
    # 'train_label_file' : "/data/nus_21class/new_data/k_8/train_labels_repeat10.mat",
    # 'images_file' : "/data/nus_21class/new_data/k_8/new_train_image.mat",

    # # 'textaug_file': "/data/nus_21class/new_data/k_8/aug_text_feature.mat",
    # 'textaug_file': "/data/nus_21class/new_data/k_8/augtext_8_top20.mat",
    # # 'textaug_file': "/data/nus_21class/new_data/k_8/augtext_8_last20.mat",
    # # 'imgaug_file': "/data/nus_21class/new_data/k_8/aug_image_feature.mat",
    # 'imgaug_file': "/data/nus_21class/new_data/k_8/aug_8_top20.mat",
    # # 'imgaug_file': "/data/nus_21class/new_data/k_8/aug_8_last20.mat",
    #TODO:K=16
    # 'text_file' : "/data/nus_21class/new_data/k_16/new_train_text.mat",
    # 'train_label_file' : "/data/nus_21class/new_data/k_16/train_labels_repeat10.mat",
    # 'images_file' : "/data/nus_21class/new_data/k_16/new_train_image.mat",

    # # 'textaug_file': "/data/nus_21class/new_data/k_16/aug_text_feature.mat",
    # 'textaug_file': "/data/nus_21class/new_data/k_16/augtext_16_top20.mat",
    # # 'textaug_file': "/data/nus_21class/new_data/k_16/augtext_16_last20.mat",

    # # 'imgaug_file': "/data/nus_21class/new_data/k_16/aug_image_feature.mat",
    # 'imgaug_file': "/data/nus_21class/new_data/k_16/aug_16_top20.mat",
    # # 'imgaug_file': "/data/nus_21class/new_data/k_16/aug_16_last20.mat",

    'text_testfile': "/data/nus_21class/new_test_text.mat",
    'image_testfile': "/data/nus_21class/new_test_image.mat",
    'test_labelfile': '/data/nus_21class/test_label.mat',
    'text_retrievalfile': "/data/nus_21class/new_retrieval_text.mat",
    'images_retrievalfile': "/data/nus_21class/new_retrieval_image.mat",
    'retrieval_labelfile': '/data/nus_21class/retrieval_label.mat'

}
dataset_lite = namedtuple('dataset_lite', ['img_feature', 'txt_feature', 'label'])
train_dataset_lite = namedtuple('train_dataset_lite', ['img_feature', 'img_augfeature','txt_feature', 'txt_augfeature', 'pooled_average_txtfeature', 'pooled_average_imgfeature','label'])

def load_coco(mode):
    if mode == 'train':
# 加载并转换特征数据
        img_feature = torch.tensor(loadmat(paths_COCO['images_file'])['features'], dtype=torch.float32)  # [num_img_features, D]
        img_augfeature = torch.tensor(loadmat(paths_COCO['imgaug_file'])['features'], dtype=torch.float32)  # [num_img_aug_features, D]
        txt_feature = torch.tensor(loadmat(paths_COCO['text_file'])['features'], dtype=torch.float32)  # [num_txt_features, D]
        txt_augfeature = torch.tensor(loadmat(paths_COCO['textaug_file'])['features'], dtype=torch.float32)  # [num_txt_aug_features, D]
        
        #TODO:mir和nus
        label = torch.tensor(loadmat(paths_COCO['train_label_file'])['labels'], dtype=torch.float32)
        # TODO:coco
        # label = loadmat(paths_COCO['train_label_file'])['labels']
        # label = torch.tensor(label[:,1:].astype(np.float32)) #TODO:coco数据集需要加上，query和retrieval也需要

        # label = np.repeat(label, 2, axis=0)
        # label = label[indices,:]

        #TODO：策略1：1+19式堆叠
        num_txt_features = txt_feature.shape[0]
        num_img_features = img_feature.shape[0]
        new_txt_features = []
        new_augtxt_features = []
        new_img_features = []
        new_augimg_features = []

        for i in range(num_txt_features):
            new_txt_features.append(txt_feature[i])
            start_idx1 = i * 20
            end_idx1 = (i * 20) + 9
            start_idx2 = (i * 20) + 10
            end_idx2 = (i * 20) + 20
            new_txt_features.extend(txt_augfeature[start_idx1:end_idx1])
            new_augtxt_features.extend(txt_augfeature[start_idx2:end_idx2])
        for i in range(num_img_features):
            new_img_features.append(img_feature[i])
            start_idx1 = i * 20
            end_idx1 = (i * 20) + 9
            start_idx2 = (i * 20) + 10
            end_idx2 = (i * 20) + 20
            new_img_features.extend(img_augfeature[start_idx1:end_idx1])
            new_augimg_features.extend(img_augfeature[start_idx2:end_idx2])
        # 将列表堆叠成torch.Tensor
        new_txt_features = torch.stack(new_txt_features)  # [num_txt_features * 10, D]
        new_augtxt_features = torch.stack(new_augtxt_features)  # [num_txt_features * 10, D]
        new_img_features = torch.stack(new_img_features)  # [num_img_features * 10, D]
        new_augimg_features = torch.stack(new_augimg_features)

        pooled_average_txtfeature = []
        for i in range(num_img_features):
            txt_chunk = new_txt_features[i*10 : i*10 + 10]
            augtxt_chunk = new_augtxt_features[i*10 : i*10 + 10]

            combined_txt_tensor = torch.cat([txt_chunk, augtxt_chunk], dim=0)  # [20,512]
            avg_txt_tensor = combined_txt_tensor.mean(dim=0)  # [512]
            repeated_txt_avg = avg_txt_tensor.unsqueeze(0).repeat(10,1)
            pooled_average_txtfeature.append(repeated_txt_avg)

        pooled_average_txtfeature = torch.cat(pooled_average_txtfeature, dim=0)
        # 同理图像侧
        pooled_average_imgfeature = []
        for i in range(num_img_features):
            img_chunk = new_img_features[i*10 : i*10 + 10]
            augimg_chunk = new_augimg_features[i*10 : i*10 + 10]

            combined_img_tensor = torch.cat([img_chunk, augimg_chunk], dim=0)  # [20,512]
            avg_img_tensor = combined_img_tensor.mean(dim=0)  # [512]
            repeated_img_avg = avg_img_tensor.unsqueeze(0).repeat(10,1)
            pooled_average_imgfeature.append(repeated_img_avg)
        pooled_average_imgfeature = torch.cat(pooled_average_imgfeature, dim=0)

        return train_dataset_lite(new_img_features, new_augimg_features, new_txt_features, new_augtxt_features, pooled_average_txtfeature, pooled_average_imgfeature, label) 
        # return train_dataset_lite(new_img_features, new_augimg_features, new_txt_features, new_augtxt_features, label) 
        # return train_dataset_lite(img_feature, img_augfeature, txt_feature, txt_augfeature, label) 
        # return train_dataset_lite(img_augfeature, img_feature, txt_augfeature, txt_feature, label) 


    elif mode == 'query':
        img_feature = loadmat(paths_COCO['image_testfile'])['features']
        txt_feature = loadmat(paths_COCO['text_testfile'])['features']
        label = loadmat(paths_COCO['test_labelfile'])['labels']
        # label = label[:,1:].astype(np.float32) 

        return dataset_lite(img_feature, txt_feature, label)

    else:
        img_feature = loadmat(paths_COCO['images_retrievalfile'])['features']
        txt_feature = loadmat(paths_COCO['text_retrievalfile'])['features']
        label = loadmat(paths_COCO['retrieval_labelfile'])['labels']
        # label = label[:,1:].astype(np.float32)  

        return dataset_lite(img_feature, txt_feature, label)
    
class my_traindataset(data.Dataset):
    def __init__(self, img_feature, img_augfeature, txt_feature, txt_augfeature, pooled_average_txtfeature, pooled_average_imgfeature, label):
        self.img_feature = torch.Tensor(img_feature)
        self.img_augfeature = torch.Tensor(img_augfeature)
        self.txt_feature = torch.Tensor(txt_feature)
        self.txt_augfeature = torch.Tensor(txt_augfeature)
        self.label = torch.Tensor(label)
        self.length = self.img_feature.size(0)
        self.pooled_average_txtfeature = torch.tensor(pooled_average_txtfeature)
        self.pooled_average_imgfeature = torch.tensor(pooled_average_imgfeature)

    def __getitem__(self, item):
        return item, self.img_feature[item, :], self.img_augfeature[item, :], self.txt_feature[item, :], self.txt_augfeature[item, :], self.pooled_average_txtfeature[item,:], self.pooled_average_imgfeature[item,:],self.label[item, :]

    def __len__(self):
        return self.length
class my_dataset(data.Dataset):
    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = torch.Tensor(img_feature)
        self.txt_feature = torch.Tensor(txt_feature)
        self.label = torch.Tensor(label)
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return item, self.img_feature[item, :], self.txt_feature[item, :], self.label[item, :]

    def __len__(self):
        return self.length