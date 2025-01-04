import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from scipy.io import savemat
from scipy.io import loadmat
from scipy.spatial.distance import cosine
import pandas as pd
import os
import csv
device = 'cuda:3'


clip_model = CLIPModel.from_pretrained("/data/WangZeQun/models/clip-vit-base-patch32/")
processor_clip = CLIPProcessor.from_pretrained("/data/WangZeQun/models/clip-vit-base-patch32/")
clip_model = clip_model.to(device)

processor_blip = BlipProcessor.from_pretrained("/data/WangZeQun/models/blip-image-captioning-large/")
model = BlipForConditionalGeneration.from_pretrained("/data/WangZeQun/models/blip-image-captioning-large/").to(device)

def extract_features_and_return_mat(keys, image_folder_path):
    features = []
    for i in range(20):
        image_path = os.path.join(image_folder_path,str(keys) +'_'+str(i)+'.jpg')
        image = Image.open(image_path)
        image = processor_clip(images=image, return_tensors="pt")
        image.data['pixel_values']=image.data['pixel_values'].to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**image)
            features.append(img_features.cpu().numpy())     
    print(f'{keys}处理完成') 

    # 将特征数组转换为一个大的Numpy数组
    features_array = np.vstack(features)
    return features_array

def extract_features_and_return_mat1(keys, image_folder_path):
    features = []
    for i in range(60):
        image_path = os.path.join(image_folder_path,str(keys) +'_'+str(i)+'.jpg')
        image = Image.open(image_path)
        image = processor_clip(images=image, return_tensors="pt")
        image.data['pixel_values']=image.data['pixel_values'].to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**image)
            features.append(img_features.cpu().numpy())     
    print(f'{keys}处理完成') 

    # 将特征数组转换为一个大的Numpy数组
    features_array = np.vstack(features)
    return features_array
    # 保存到.mat文件
    # savemat(output_mat_file, {'features': features_array})

def extract_orifeatures_and_return_mat(keys, image_folder_path):
            
    features = []
    image_path = os.path.join(image_folder_path, str(keys) +'.jpg')
    image = Image.open(image_path)
    image = processor_clip(images=image, return_tensors="pt")
    image.data['pixel_values']=image.data['pixel_values'].to(device)
    with torch.no_grad():
        img_features = clip_model.get_image_features(**image)
        features.append(img_features.cpu().numpy())     
    print(f'{keys}处理完成') 

    # 将特征数组转换为一个大的Numpy数组
    features_array = np.vstack(features)
    return features_array

def read_indices_from_file(input_file):
    """从文件中读取索引列表"""
    indices = {}
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            if row:  # 确保行不为空
                index, image_id = i, row[0]
                indices[index] = image_id
    return indices

def generate_captions_for_images(source_folder, aug_folder, model, device='cuda'):
    results = {"idx": [], "captions": []}

    for filename in os.listdir(source_folder):
        for i in range(20):
            file_path = os.path.join(aug_folder, f"{filename}_{i}.jpg")
            img = Image.open(file_path).convert('RGB')

            # 使用 BLIP 模型生成描述
            inputs = processor_blip(images=img, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            output = processor_blip.decode(out[0], skip_special_tokens=True)
            print(output)

            # 保存结果
            filename1 = f"{filename.split('.')[0]}_{i}.jpg"
            results["idx"].append(filename1)
            results["captions"].append(output)

    return results

prefix = '2'
# 加载键
# "/data/WangZeQun/COCO-2017/few_shot_final/k_16/aug_train_image_16shot/"
oriimage_folder_path = "/data/WangZeQun/nus_21class/nus_raw/raw_nus/images"
image_folder_path_1 = f"/data/WangZeQun/nus_21class/new_data/k_{prefix}/aug_image/"
image_folder_path_2 = f"/data/WangZeQun/nus_21class/new_data/k_{prefix}/aug_{prefix}_20-40/"
output_mat_file_top = f'/data/WangZeQun/nus_21class/new_data/k_{prefix}/aug_{prefix}_top20.mat'
output_mat_file_last = f'/data/WangZeQun/nus_21class/new_data/k_{prefix}/aug_{prefix}_last20.mat'
output_textmat_file_top = f'/data/WangZeQun/nus_21class/new_data/k_{prefix}/augtext_{prefix}_top20.mat'
output_textmat_file_last = f'/data/WangZeQun/nus_21class/new_data/k_{prefix}/augtext_{prefix}_last20.mat'
train_indices_file = f"/data/WangZeQun/nus_21class/new_data/k_{prefix}/train_keys.npy"
output_csv_file_top = f'/data/WangZeQun/nus_21class/new_data/k_{prefix}/aug_{prefix}_top20.csv'
output_csv_file_last = f'/data/WangZeQun/nus_21class/new_data/k_{prefix}/aug_{prefix}_last20.csv'
keys = np.load(train_indices_file)

# 初始化一个空列表，用于保存所有筛选后的特征
top_features_array = []
last_features_array = []
top_caption = {"idx": [], "captions": []}
last_caption = {"idx": [], "captions": []}
top_textfeatures_array = []
last_textfeatures_array = []
for i, key in enumerate(keys):
    origin_data = extract_orifeatures_and_return_mat(key, oriimage_folder_path)
    data1 = extract_features_and_return_mat(key, image_folder_path_1)
    data2 = extract_features_and_return_mat1(key, image_folder_path_2)

    # 将 origin_data 复制为相同数量的向量 (20) 用于计算相似度
    origin_data_repeated = np.repeat(origin_data, 20, axis=0)  # shape: (20, 512)
    origin_data_repeated1 = np.repeat(origin_data, 60, axis=0)  # shape: (20, 512)
    # 计算 origin_data 与 data1 和 data2 的余弦相似度
    cos_sim_data1 = 1 - np.array([cosine(origin_data_repeated[j], data1[j]) for j in range(20)])  # shape: (20,)
    cos_sim_data2 = 1 - np.array([cosine(origin_data_repeated1[j], data2[j]) for j in range(60)])  # shape: (20,)
    # 将 data1 和 data2 的特征进行合并，以及对应的相似度
    all_data = np.vstack((data1, data2))  # shape: (40, 512)
    all_cos_sim = np.concatenate((cos_sim_data1, cos_sim_data2))  # shape: (40,)

    # 找出前 20 个最相似的向量索引
    top_20_indices = np.argsort(all_cos_sim)[-20:]
    #在这里补充遍历索引，然后根据索引如果小于20则使用image_folder_path_1来读取图像来生成对应的文本，如果大于20则image_folder_path_2来读取

    #后二十个索引
    last_20_indices = np.argsort(all_cos_sim)[:20]

    # 选择相应的前 20 个最相似的特征向量
    top_20_features = all_data[top_20_indices]
    last_20_features = all_data[last_20_indices]

    # 将选出来的特征添加到 features_array 列表中
    top_features_array.append(top_20_features)
    last_features_array.append(last_20_features)

    for idx in top_20_indices:
        if idx < 20:
            file_path = os.path.join(image_folder_path_1, f"{key}_{idx}.jpg")
        else:
            file_path = os.path.join(image_folder_path_2, f"{key}_{idx - 20}.jpg")
        img = Image.open(file_path).convert('RGB')
        inputs = processor_blip(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        output = processor_blip.decode(out[0], skip_special_tokens=True)
        top_caption["idx"].append(f"{key}_{idx}.jpg")
        top_caption["captions"].append(output)

        # 提取文本特征
        tokenized_template_clazz = processor_clip.tokenizer(output, padding=True, truncation=True, return_tensors="pt", max_length=77)
        tokenized_template_clazz['input_ids'] = tokenized_template_clazz['input_ids'].to(device)
        tokenized_template_clazz['attention_mask'] = tokenized_template_clazz['attention_mask'].to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**tokenized_template_clazz)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        top_textfeatures_array.append(text_features.squeeze().cpu().numpy())

    for idx in last_20_indices:
        if idx < 20:
            file_path = os.path.join(image_folder_path_1, f"{key}_{idx}.jpg")
        else:
            file_path = os.path.join(image_folder_path_2, f"{key}_{idx - 20}.jpg")
        img = Image.open(file_path).convert('RGB')
        inputs = processor_blip(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        output = processor_blip.decode(out[0], skip_special_tokens=True)
        last_caption["idx"].append(f"{key}_{idx}.jpg")
        last_caption["captions"].append(output)

        # 提取文本特征
        tokenized_template_clazz = processor_clip.tokenizer(output, padding=True, truncation=True, return_tensors="pt", max_length=77)
        tokenized_template_clazz['input_ids'] = tokenized_template_clazz['input_ids'].to(device)
        tokenized_template_clazz['attention_mask'] = tokenized_template_clazz['attention_mask'].to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**tokenized_template_clazz)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        last_textfeatures_array.append(text_features.squeeze().cpu().numpy())

# 将 features_array 列表转换为 numpy 数组以保存
top_features_array = np.vstack(top_features_array)  # 将所有 key 的结果垂直合并，最终形状为 (num_keys * 20, 512)
last_features_array = np.vstack(last_features_array)  # 将所有 key 的结果垂直合并，最终形状为 (num_keys * 20, 512)
top_textfeatures_array = np.vstack(top_textfeatures_array)  # 将所有 key 的结果垂直合并，最终形状为 (num_keys * 20, 512)
last_textfeatures_array = np.vstack(last_textfeatures_array)  # 将所有 key 的结果垂直合并，最终形状为 (num_keys * 20, 512)

# 保存为 .mat 文件
savemat(output_mat_file_top, {'features': top_features_array})
savemat(output_mat_file_last, {'features': last_features_array})
savemat(output_textmat_file_top, {'features': top_textfeatures_array})
savemat(output_textmat_file_last, {'features': last_textfeatures_array})

caption_top = pd.DataFrame(top_caption)
caption_last = pd.DataFrame(last_caption)
caption_top.to_csv(output_csv_file_top, index=False)
caption_last.to_csv(output_csv_file_last, index=False)
