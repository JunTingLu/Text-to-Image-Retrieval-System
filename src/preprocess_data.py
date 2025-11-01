"""
「tex_search_img.ipynb」的副本
"""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
# from ultralytics import YOLO
# import cv2
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import re
from typing import List, Optional
import csv
import json
from googletrans import Translator
import asyncio

# Define the image transformation
def preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # 常態分布 (RGB:0-255)
    ])
    
# Custom Dataset for fine-tuning
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, transform=None): # transform including resize, crop,..etc
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
        
    def __len__(self):return len(self.image_paths)
    # magic method，用來讓物件支援像「索引」一樣的存取行為，例如 obj[index]
    def __getitem__(self, idx): 
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        if self.transform:
            image = self.transform(image)
        return image, text

    # consider ViT-B/32 with input 224x224
    def resize_image(self, image:Image.Image, size = (224,224)): 
        return image.resize(size)
    
    # 拼接圖片
    def image_collage(self, out_dir="./result", thumb_width:int=300,  thumb_heigh:int=300):
        thumb_size = (thumb_width, thumb_heigh)
        num_thumb = 2
        # 每四張拼接一次
        for batch_idx in range(0, len(self.image_paths), num_thumb**2):
            batch = self.image_paths[batch_idx : batch_idx + num_thumb**2]
            bg = Image.new("RGB", (thumb_size[0] * num_thumb, thumb_size[1] * num_thumb), "#000000")
            for j, path in enumerate(batch):
                img = Image.open(path).convert("RGB")
                resize_img = self.resize_image(img, thumb_size)  # 假設 resize_image 回傳 PIL Image
                x = (j % 2) * thumb_size[0]
                y = (j // 2) * thumb_size[1]
                bg.paste(resize_img, (x, y))
            out_path = os.path.join(out_dir, f"concat_{batch_idx//4}.jpg")
            bg.save(out_path)
            print("saving collage successfully!")

class FlickrDataset(Dataset):
    def __init__(self, img_list, txt_list, transform):
        self.imgs, self.txts, self.tf = img_list, txt_list, transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img = self.tf(img)
        return img, self.txts[idx] 
    
# 將數據整理成一張圖像對三種不同 caption JSON 格式
def aggregate_image_captions(image_path:str, captions:str):
    """
    Inputs:
        captions = "1. a man riding a bike\n2. a cyclist wearing a helmet\n3. an outdoor biking scene"
    Returns:
        images = ["img.jpg", "img.jpg", "img.jpg"]
        captions_lst = [
            "a man riding a bike",
            "a cyclist wearing a helmet",
            "an outdoor biking scene"
        ]
    """
    # 使用正則表達式“切分”，依據換行符號或數字標號 (1. 2. 3.)，被切分的部分可能留下 ''
    split_captions = re.split(r'(?:\d+\s*[\.\)]\s*|\n+)', captions)
    # 去除空白與空字串
    captions_lst = [cap.strip() for cap in split_captions if cap.strip()]
    # 複製 image_path，數量與 captions 一致
    images = [image_path] * len(captions_lst)
    return images, captions_lst

def expand_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 df 中的 caption 拆解為多筆
    """
    expanded_rows = []
    for _,row in df.iterrows():
        # print(row["Caption"])
        imgs, caps = aggregate_image_captions(row["image_path"], row["Caption"])
        # caps represents a list of muti-captions
        for img, cap in zip(imgs, caps):
            # transfer into pair of single image_path with single caption
            expanded_rows.append({"image_path": img, "Caption": cap})
    return pd.DataFrame(expanded_rows)    

# 移除caption中重複的文字
def deduplicate_contxt(text: str, sep: str = "，"):
    """
    精確句子去重
    :param text: 原始文字
    :param sep: 切分符號，默認是 "。" (適用中文)
    :return: 去重後的文字
    """
    # 切分句子
    sentences = text.split(sep)
    # 去掉空字串並保留順序去重
    unique_sentences = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))
    # 拼回文字
    return sep.join(unique_sentences) + (sep if text.endswith(sep) else "")

async def load_opendataset(
            img_dir="../crossmodal-3600/images", 
            text_dir="../crossmodal-3600/captions.jsonl", 
            output_csv="open_data.csv",
        ):
        """
        讀取 jsonl 並將 caption 翻譯 + 潤飾後輸出為 CSV，並行加速版本
        Args:
            img_dir (str, optional): _description_. Defaults to "../crossmodal-3600/images".
            text_dir (str, optional): _description_. Defaults to "../crossmodal-3600/captions.jsonl".
            output_csv (str, optional): _description_. Defaults to "open_data.csv".
        """
        records = []
        # 1. 讀取 captions jsonl
        with open(text_dir, "r", encoding="utf-8") as f:
            for line in (f):
                data = json.loads(line)
                image_path = f"{img_dir}/{data.get("image/key")}.jpg"
                caption = data.get("zh").get("caption")
                # print(37, caption)
                n_img = len(caption)
                image_paths = [image_path] * n_img
                for img, cap in zip(image_paths, caption):
                    records.append([img, cap])        
        # 2. 使用 asyncio.gather 進行並行翻譯
        tasks = [process_caption(img, cap) for img, cap in records]
        results = await asyncio.gather(*tasks)
        # 3. 寫入 CSV
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "Caption"])
            writer.writerows(results)
            print(f"✅ 已完成！共寫入 {len(results)} 筆資料到 {output_csv}")
        return output_csv
            
async def process_caption(img, cap, semaphore = asyncio.Semaphore(5)):
    """
    asyncio.Semaphore(5) : 控制同時併發的翻譯請求數量（例如一次最多 5 筆）
    """
    async with semaphore:
        async with Translator(timeout=15) as translator:
            """翻譯單句 caption 為繁體中文"""
            # 1. 翻譯
            translated = await translator.translate(cap, dest='zh-tw')
            print(translated)
            return [img, translated.text]
                
# 使用 yolo8 裁切圖片主體             
# def crop_objects(
#     img_paths,
#     model_path="yolov8l.pt",
#     output_dir="../cropped_objects",
#     conf_threshold=0.8,
#     min_area=224*224,
#     max_area=1024*1024,
#     aspect_ratio_min=0.5,
#     aspect_ratio_max=2.0,
#     per_image_cap=5
#     ):
#     """
#     使用 YOLOv8 模型偵測並裁切圖片中的物體，輸出到指定資料夾。

#     Args:
#         img_paths (list or str): 單張圖片路徑 (str) 或多張圖片路徑 (list)。
#         model_path (str): YOLOv8 模型路徑，預設使用官方輕量模型。
#         output_dir (str): 輸出裁切物體的資料夾。

#     Returns:
#         dict: {圖片路徑: [裁切後圖片路徑列表]}
#     """
#     # 確保輸出資料夾存在
#     os.makedirs(output_dir, exist_ok=True)
#     # 載入 YOLO 模型
#     model = YOLO(model_path)
#     # 若輸入單張圖片，自動轉成 list
#     if isinstance(img_paths, str):
#         img_paths = [img_paths]
        
#     result_dict = {}
#     print(144, img_paths)
#     for img_path in img_paths:
#         results = model(img_path)
#         # print(115, f"預設有 80 類別 (COCO dataset):{model.names}")
#         img = cv2.imread(img_path)
#         cropped_files = []
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             obj_img = img[y1:y2, x1:x2]
#             out_path = os.path.join(
#                 output_dir,
#                 f"{os.path.splitext(os.path.basename(img_path))[0]}_obj_{i}.jpg"
#             )
#             cv2.imwrite(out_path, obj_img)
#             cropped_files.append(out_path)
#         result_dict[img_path] = cropped_files
#     return result_dict


# 處理格式為 txt 資料，包含 image_path / caption
def load_txt_to_dataframe(txt_path: str) -> pd.DataFrame:
    """將 txt 資料轉換成 DataFrame（格式：image_path, caption）"""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # 跳過第一行
    data = []
    for line in lines:
        parts = line.strip().split(",", 1)  # 只分割第一個逗號
        if len(parts) == 2:
            image, caption = parts
            data.append([image.strip(), caption.strip()])
    return pd.DataFrame(data, columns=["image_path", "Caption"])


# 處理格式為 csv 資料，包含 image_path / caption
def load_and_merge_csv_files(csv_paths: List[str], aggregate: bool = False) -> pd.DataFrame:
    """將多個 CSV 合併成單一 DataFrame，並可選擇是否對 caption 進行聚合處理"""
    dfs = []
    for path in csv_paths:
        if Path(path).is_file():
            df = pd.read_csv(path, encoding="utf-8-sig")
            df = df.dropna(subset=["Caption"])
            if aggregate:
                df = expand_dataframe(df)  # 自定義 caption 聚合函式
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("沒有找到可用的 CSV 檔案。")
    return pd.concat(dfs, ignore_index=True, axis=0)

# 整合數據 (txt or csv)
def process_dataset(
        txt_file: Optional[str] = None,
        csv_files: Optional[List[str]] = None,
        output_file: str = "./merged_dataset.csv",
        aggregate_caption: bool = False
    ) -> pd.DataFrame:
    """
    依據檔案類型自動處理資料（txt 或多個 csv）並輸出合併結果

    Args:
        txt_file (str, optional): 若數據集為 txt 格式，包含 image_path, caption。
        csv_files (List[str], optional): 多個 CSV 檔案路徑。
        output_file (str): 輸出合併後的檔案名稱。
        translate (bool): 是否啟用翻譯功能（預留擴充）。
        aggregate_caption (bool): 是否針對 caption 重組含有 1,2,3 數字前綴的 caption。
    """
    # Step 1: 載入資料
    if txt_file:
        print(f"偵測到 TXT 檔案，開始解析 {txt_file} ...")
        df_final = load_txt_to_dataframe(txt_file)
    elif csv_files and len(csv_files) >= 2:
        print(f"偵測到 {len(csv_files)} 個 CSV，開始自動合併 ...")
        df_final = load_and_merge_csv_files(csv_files, aggregate=aggregate_caption)
    elif len(csv_files) == 1:
        df= pd.read_csv(csv_files[0], encoding="utf-8-sig")
        # 移除 NaN 列
        df = df.dropna(subset=["Caption"])
        df_final = expand_dataframe(df)
    else:
        raise ValueError("請提供 txt_file 或至少兩個 csv_files 進行合併。")
    # Step 2: 輸出合併後結果
    df_final.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ 已輸出合併後檔案：{output_file}（共 {len(df_final)} 筆資料）")
    return output_file


# if __name__=="__main__":
    # command "yolo settings" 可查看目前參數設定
    # print(crop_objects("origin_dataset/13274_632087950224364_8692510268907875417_n(1).jpg"))
    # df=pd.read_csv("captions_0924.csv")