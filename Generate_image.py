import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import pandas as pd
from scipy.io import loadmat
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("/models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

save_folder = "/data/augmented_image/" # 保存生成图像的目标文件夹路径

os.makedirs(save_folder, exist_ok=True)  # 确保目标文件夹存在

csv_file = "/data/Blip_generate.csv" #TODO:通过BLIP生成文本后加上label信息的文本
captions = pd.read_csv(csv_file)
captions.set_index('idx',inplace=True)

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

def generate_filename(base_filename, index):
    name_part, extension = base_filename.rsplit('.', 1)
    return f"{name_part}_{index}.{'jpg'}"
# 遍历源文件夹中的每个文件
for index, row in captions.iterrows():

    result_path_positive = []
    for i in range(40):
        result_filename = generate_filename(index, i+20)
        result_path = os.path.join(save_folder, result_filename)
        result_path_positive.append(result_path)

    text = [captions.loc[index,"captions"]]
 
    # 生成图像并保存
    with torch.no_grad():
        for i, result_path in enumerate(result_path_positive):
            if not os.path.exists(result_path):
                seed = i+20  
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
