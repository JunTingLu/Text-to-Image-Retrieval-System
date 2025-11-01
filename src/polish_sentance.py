from tqdm import tqdm
from googletrans import Translator
import ollama
import json
import csv
import asyncio

semaphore = asyncio.Semaphore(5)

async def load_opendataset(
        img_dir="../crossmodal-3600/images", 
        text_dir="../crossmodal-3600/captions.jsonl", 
        output_csv="open_data.csv"
    ):
    """
    讀取 jsonl 並將 caption 翻譯 + 潤飾後輸出為 CSV，並行加速版本
    Args:
        img_dir (str, optional): _description_. Defaults to "../crossmodal-3600/images".
        text_dir (str, optional): _description_. Defaults to "../crossmodal-3600/captions.jsonl".
        output_csv (str, optional): _description_. Defaults to "open_data.csv".
    """
    processed_records = []
    records = []
    # 1. 讀取 captions jsonl
    with open(text_dir, "r", encoding="utf-8") as f:
        for line in (f):
            data = json.loads(line)
            image_path = img_dir + data.get("image/key") + ".jpg"
            caption = data.get("zh").get("caption")
            # print(37, caption)
            n_img = len(caption)
            image_paths = [image_path] * n_img
            for img, cap in zip(image_paths, caption):
                records.append([img, cap])
            
    # lines = [json.loads(line.strip()) for idx, line in enumerate(f) if idx < max_images]
    
    # 2. 使用 asyncio.gather 進行並行翻譯
    tasks = [process_caption(img, cap) for img, cap in records]
    results = await asyncio.gather(*tasks)

    # 3. 寫入 CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "Caption"])
        writer.writerows(results)
        print(f"✅ 已完成！共寫入 {len(results)} 筆資料到 {output_csv}")


async def process_caption(img, cap):
    async with semaphore:
        async with Translator(timeout=15) as translator:
            """翻譯 + 潤飾 單句 caption"""
            # 1. 翻譯
            translated = await translator.translate(cap, dest='zh-tw')
            print(translated)
            # 2. 潤飾
            # response = ollama.chat(
            #     model="qwen2:7b-instruct",  # 用輕量模型加速
            #     messages=[{
            #         "role": "user",
            #         "content": f"請將以下句子修飾成自然流暢的繁體中文：\n\n{translated}",
            #     }],
            #     options={"temperature": 0}
            # )
            return [img, translated.text]


if __name__=="__main__":         
    asyncio.run(load_opendataset())