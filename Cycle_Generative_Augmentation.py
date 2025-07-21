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
from diffusers import DiffusionPipeline
import csv
device = 'cuda:3'
pipe = DiffusionPipeline.from_pretrained("/models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

clip_model = CLIPModel.from_pretrained("/data/models/clip-vit-base-patch32/")
processor_clip = CLIPProcessor.from_pretrained("/data/models/clip-vit-base-patch32/")
clip_model = clip_model.to(device)

processor_blip = BlipProcessor.from_pretrained("/data/models/blip-image-captioning-large/")
model = BlipForConditionalGeneration.from_pretrained("/data/models/blip-image-captioning-large/").to(device)

negative_prompts = [
    ["distorted, sketchy, caricature, rough, grotesque, unfinished, cartoonish"],
    ["fuzzy, warped, animated, outline, unpleasant, preliminary, cartoon-like"],
    ["unclear, misshapen, comic, doodle, unattractive, initial, cartoonish"],
    ["cloudy, malformed, manga, trace, hideous, sketch, cartoon-like"],
    ["obscure, twisted, stick figure, blueprint, unsightly, basic, cartoonish"],
    ["hazy, contorted, childish, diagram, foul, prototype, cartoon-like"],
    ["vague, irregular, pixelated, schematic, repulsive, rough draft, cartoonish"],
    ["indistinct, deformed, abstract, plan, offensive, early version, cartoon-like"],
    ["muddy, abnormal, fantasy, layout, unappealing, concept, cartoonish"],
    ["smudged, misproportioned, fictional, pattern, grotesque, formative, cartoon-like"],
    ["blurry, distorted, animated, outline, unattractive, preliminary, cartoonish"],
    ["unclear, warped, comic, sketch, unpleasant, initial, cartoon-like"],
    ["cloudy, misshapen, manga, doodle, hideous, early version, cartoonish"],
    ["obscure, malformed, stick figure, trace, unsightly, rough draft, cartoon-like"],
    ["hazy, twisted, childish, blueprint, foul, sketch, cartoonish"],
    ["vague, contorted, pixelated, diagram, repulsive, concept, cartoon-like"],
    ["indistinct, irregular, abstract, schematic, offensive, formative, cartoonish"],
    ["muddy, deformed, fantasy, plan, unappealing, prototype, cartoon-like"],
    ["smudged, abnormal, fictional, layout, grotesque, basic, cartoonish"],
    ["blurry, misproportioned, caricature, pattern, unsightly, unfinished, cartoon-like"],
    ["distorted, sketchy, caricature, rough, grotesque, unfinished, cartoonish"],
    ["fuzzy, warped, animated, outline, unpleasant, preliminary, cartoon-like"],
    ["unclear, misshapen, comic, doodle, unattractive, initial, cartoonish"],
    ["cloudy, malformed, manga, trace, hideous, sketch, cartoon-like"],
    ["obscure, twisted, stick figure, blueprint, unsightly, basic, cartoonish"],
    ["hazy, contorted, childish, diagram, foul, prototype, cartoon-like"],
    ["vague, irregular, pixelated, schematic, repulsive, rough draft, cartoonish"],
    ["indistinct, deformed, abstract, plan, offensive, early version, cartoon-like"],
    ["muddy, abnormal, fantasy, layout, unappealing, concept, cartoonish"],
    ["smudged, misproportioned, fictional, pattern, grotesque, formative, cartoon-like"],
    ["blurry, distorted, animated, outline, unattractive, preliminary, cartoonish"],
    ["unclear, warped, comic, sketch, unpleasant, initial, cartoon-like"],
    ["cloudy, misshapen, manga, doodle, hideous, early version, cartoonish"],
    ["obscure, malformed, stick figure, trace, unsightly, rough draft, cartoon-like"],
    ["hazy, twisted, childish, blueprint, foul, sketch, cartoonish"],
    ["vague, contorted, pixelated, diagram, repulsive, concept, cartoon-like"],
    ["indistinct, irregular, abstract, schematic, offensive, formative, cartoonish"],
    ["muddy, deformed, fantasy, plan, unappealing, prototype, cartoon-like"],
    ["smudged, abnormal, fictional, layout, grotesque, basic, cartoonish"],
    ["blurry, misproportioned, caricature, pattern, unsightly, unfinished, cartoon-like"]
]

def extract_features_and_return_mat(keys, image_folder_path):
    features = []
    for i in range(80):
        image_path = os.path.join(image_folder_path,str(keys) +'_'+str(i)+'.jpg')
        image = Image.open(image_path)
        image = processor_clip(images=image, return_tensors="pt")
        image.data['pixel_values']=image.data['pixel_values'].to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**image)
            features.append(img_features.cpu().numpy())     
    print(f'{keys}finished') 

    features_array = np.vstack(features)
    return features_array


def extract_orifeatures_and_return_mat(keys, image_folder_path):
            
    features = []
    image_path = os.path.join(image_folder_path, str(keys) +'.jpg')
    image = Image.open(image_path)
    image = processor_clip(images=image, return_tensors="pt")
    image.data['pixel_values']=image.data['pixel_values'].to(device)
    with torch.no_grad():
        img_features = clip_model.get_image_features(**image)
        features.append(img_features.cpu().numpy())     
    print(f'{keys}finished') 

    features_array = np.vstack(features)
    return features_array

def read_indices_from_file(input_file):
   
    indices = {}
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            if row: 
                index, image_id = i, row[0]
                indices[index] = image_id
    return indices

def generate_captions_for_images(source_folder, aug_folder, model, device='cuda'):
    results = {"idx": [], "captions": []}

    for filename in os.listdir(source_folder):
        for i in range(80):
            file_path = os.path.join(aug_folder, f"{filename}_{i}.jpg")
            img = Image.open(file_path).convert('RGB')

            inputs = processor_blip(images=img, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            output = processor_blip.decode(out[0], skip_special_tokens=True)
            print(output)

            filename1 = f"{filename.split('.')[0]}_{i}.jpg"
            results["idx"].append(filename1)
            results["captions"].append(output)

    return results


prefix = '2'
oriimage_folder_path = "/data/nus_21class/nus_raw/raw_nus/images"
image_folder_path_1 = f"/data/nus_21class/new_data/k_{prefix}/aug_image/"
output_mat_file_top = f'/data/nus_21class/new_data/k_{prefix}/aug_{prefix}_top20.mat'
output_textmat_file_top = f'/data/nus_21class/new_data/k_{prefix}/augtext_{prefix}_top20.mat'
train_indices_file = f"/data/nus_21class/new_data/k_{prefix}/train_keys.npy"
output_csv_file_top = f'/data/nus_21class/new_data/k_{prefix}/aug_{prefix}_top20.csv'
keys = np.load(train_indices_file)

save_folder = "/data/nus_21class/new_data/k_{prefix}/aug_image/" 
os.makedirs(save_folder, exist_ok=True) 

# 1step:Semantic-driven text synthesis
results = {"idx": [], "captions": []}
for filename in os.listdir(oriimage_folder_path):
    file_path = filename + "_" + '.jpg'
    img = Image.open(file_path).convert('RGB')
    inputs = processor_blip( images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    output = processor_blip.decode(out[0], skip_special_tokens=True)
    print(output)
    filename1 = filename.split('.')[0] + "_" + '.jpg'
    results["idx"].append(filename1)
    results["captions"].append(output)

caption_df = pd.DataFrame(results)
caption_df.to_csv(output_csv_file_top)

# 2step:Conditional diffusion generation
def generate_filename(base_filename, index):
    name_part, extension = base_filename.rsplit('.', 1)
    return f"{name_part}_{index}.{'jpg'}"
for index, row in results.iterrows():

    result_path_positive = []
    for i in range(80):
        result_filename = generate_filename(index, i)
        result_path = os.path.join(save_folder, result_filename)
        result_path_positive.append(result_path)

    text = [results.loc[index,"captions"]]
 
    with torch.no_grad():
        for i, result_path in enumerate(result_path_positive):
            if not os.path.exists(result_path):
                seed = i
                generator = torch.manual_seed(seed)
                positive_prompt = text
                images = pipe(prompt=text,
                              negative_prompt=negative_prompts[i],
                              generator=generator).images[0]

                images.save(result_path)
            else:
                print(f"File {result_path} already exists. Skipping image generation.")
  
    print(f"Generated {index}")

print("Processing completed for all files.")

#3 step:Inconsistency filtering
top_features_array = []
last_features_array = []
top_caption = {"idx": [], "captions": []}
last_caption = {"idx": [], "captions": []}
top_textfeatures_array = []
last_textfeatures_array = []
for i, key in enumerate(keys):
    origin_data = extract_orifeatures_and_return_mat(key, oriimage_folder_path)
    data1 = extract_features_and_return_mat(key, image_folder_path_1)

    
    origin_data_repeated = np.repeat(origin_data, 20, axis=0)  
    
    cos_sim_data1 = 1 - np.array([cosine(origin_data_repeated[j], data1[j]) for j in range(20)]) 


   
    top_20_indices = np.argsort(cos_sim_data1)[-20:]
    top_20_features = data1[top_20_indices]
    top_features_array.append(top_20_features)

    for idx in top_20_indices:

        file_path = os.path.join(image_folder_path_1, f"{key}_{idx}.jpg")
        img = Image.open(file_path).convert('RGB')
        inputs = processor_blip(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        output = processor_blip.decode(out[0], skip_special_tokens=True)
        top_caption["idx"].append(f"{key}_{idx}.jpg")
        top_caption["captions"].append(output)

        tokenized_template_clazz = processor_clip.tokenizer(output, padding=True, truncation=True, return_tensors="pt", max_length=77)
        tokenized_template_clazz['input_ids'] = tokenized_template_clazz['input_ids'].to(device)
        tokenized_template_clazz['attention_mask'] = tokenized_template_clazz['attention_mask'].to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**tokenized_template_clazz)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        top_textfeatures_array.append(text_features.squeeze().cpu().numpy())


top_features_array = np.vstack(top_features_array)  
top_textfeatures_array = np.vstack(top_textfeatures_array)  


savemat(output_mat_file_top, {'features': top_features_array})
savemat(output_textmat_file_top, {'features': top_textfeatures_array})

caption_top = pd.DataFrame(top_caption)
caption_top.to_csv(output_csv_file_top, index=False)
