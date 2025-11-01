# ====================== text to search images ======================
import numpy as np
import os, torch, clip
from PIL import Image
import glob
import matplotlib.pyplot as plt
from preprocess_data import *

model_dir = "./results/clip_ft_1031.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("Loaing model successfully!")
model.eval()
state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(state_dict)
print("✅ Fine-tuned weights loaded!")

def encode_images(img_list):
    feats = []
    with torch.no_grad():
        for p in img_list:
            img = clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f   = model.encode_image(img)
            feats.append(f.squeeze().cpu().numpy())
    feats = np.stack(feats)
    return feats / np.linalg.norm(feats, axis=1, keepdims=True) # 正規化

def search(query, img_list, img_feats, index=None, topk=5):
    with torch.no_grad():
        txt_feat = model.encode_text(clip.tokenize([query]).to(device))
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    sims = (txt_feat.cpu().numpy() @ img_feats.T).squeeze()
    # the (image, text) threshold score
    idxs = sims.argsort()[::-1][:topk] 
    return [(img_list[i], sims[i]) for i in idxs]


def display_top_images(results:list):
    """
    Args:
        results (list): 搜尋到的所有圖片列表
    """
    # 設定顯示排版
    n = len(results)
    cols = 5
    rows = n // cols + 1 if n % cols != 0 else n // cols # 自動算需要幾列
    plt.figure(figsize=(15, 5 * rows))
    for i, (path, score) in enumerate(results, 1):
        filename = path.split("/")[-1]
        img = Image.open(path)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{filename}\nscore={score:.3f}", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    # Use only the first 1000 images for indexing demo.
    folder = "../dataset_for_demo" 
    sample_imgs = glob.glob(os.path.join(folder, "*.[jp][pn]g"))  
    sample_imgs = [i for i in sample_imgs]
    sample_feats  = encode_images(sample_imgs)
    # 加入 faiss 搜尋索引
    #faiss_index = create_vector_index(sample_feats)
    query = "一個正在微笑的男人"
    search_results = search(query, sample_imgs, sample_feats, topk=5)
    for p, s in search_results:
        print(f"Here's the related image {p.split('/')[-1]:30s}, similarity score={s:.3f}")
    # 顯示所有圖片
    display_top_images(search_results)
    
    
    