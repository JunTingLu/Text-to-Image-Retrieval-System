import torch, clip
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from preprocess_data import *
from pathlib import Path
import pandas as pd
import logging
import matplotlib.pyplot as plt

saving_path = Path("results")
saving_path.mkdir(parents=True, exist_ok=True)

# 設定 logging 格式與輸出檔案
logging.basicConfig(
    filename=str(saving_path) + "/training.log", # log 檔名
    filemode="w", # "w" 覆蓋，"a" 追加
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ====================== Read-in caption  ======================
# format：'1000268201_693b08cb0e.jpg#0\tA child in a pink dress ...'
def read_caption(raw_data_path:str, is_deduplicated=False):
    """
    Args:
        data_path (str): 含有 iamge_path / caption csv 檔
        is_deduplicated (bool, optional): 決定是否針對 caption 內容去重
        is_aggregated (bool, optional): 決定是否針對 caption 重組含有 1,2,3 數字前綴的 caption
    """
    img_paths, captions = [], []
    raw_data = pd.read_csv(raw_data_path, encoding="utf-8-sig")
    # print(32, raw_data)
    # read csv with (image, text) 
    for _, raw in raw_data.iterrows():
        img, cap = raw["image_path"].strip(), raw["Caption"] 
        # 去重內容
        if is_deduplicated:
            cap = deduplicate_contxt(cap)
        # print(f"去重後的 caption \n：{cap}")
        img_paths.append(img)
        captions.append(cap)
    # print(41, img_paths)
    print(f"Total (image, caption) pairs: {len(captions)}")
    return img_paths, captions

# ====================== preprocessing dataset ======================
# consider ViT-B/32 with input 224x224
# def build_preprocess(n_px=224): 
#     return Compose([
#         Resize(n_px, interpolation=Image.BICUBIC),
#         CenterCrop(n_px),
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073),
#                   (0.26862954, 0.26130258, 0.27577711)),
#     ])

# ====================== plot the loss curve =======================
def loss_plot(num_epochs, loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='b')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(saving_path / "loss_curve.png", dpi=300)
    plt.show()

# ====================== pre-training =======================
def training(
        epochs:int, 
        model: Module, # OpenAI clip model
        optimiser: Optimizer,
        loader:DataLoader,
        device: str
    ) -> None:
    model.train()
    print("Starting fine-tuning...")
    loss_history = []
    for ep in range(epochs):
        running_loss = 0
        for imgs, txts in tqdm(loader):
            imgs = imgs.to(device)
            # truncate means 
            txts_tok = clip.tokenize(txts, truncate=True).to(device)
            logits_per_img, logits_per_txt = model(imgs, txts_tok)
            # print(f"每張圖片對所有 caption 的相似度矩陣\n {logits_per_img}")
            # 建立一個長度為 n 的一維整數序列張量，從 0 開始
            # torch.arange(5) -> tensor([0, 1, 2, 3, 4])
            tgt = torch.arange(len(imgs), device=device)
            # 計算 InfoNCE loss
            # print(63, logits_per_img.dtype)
            # 保證 logits 是 float32 if using "mps" device
            # logits_per_img = logits_per_img.to(torch.float32)
            # logits_per_txt = logits_per_txt.to(torch.float32)
            # 對於每個正樣本，batch 內的其他 N−1 個文字/圖片都當成負樣本
            loss_img = F.cross_entropy(logits_per_img, tgt)
            loss_txt = F.cross_entropy(logits_per_txt, tgt)
            loss = (loss_img + loss_txt) / 2
            # PyTorch 在 backward 時如果發現 grad=None，會直接新建一個 tensor 來存放梯度，比預設的「先建一個全零 tensor再覆寫」更省記憶體且更快
            optimiser.zero_grad(set_to_none=True) 
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            logging.info(f"Epoch [{ep+1}/{epochs}] - Loss: {running_loss:.4f}")
        avg = running_loss / len(loader)
        loss_history.append(avg)
        print(f"Epoch {ep+1}/{epochs}  loss={avg:.4f}")
        
    # ====================== save weight ======================
    torch.save(model.state_dict(), "./results/clip_ft_1031.pt")
    print("Finished!  Weights saved!")
    return loss_history


def start_tuning():
    parser = argparse.ArgumentParser(description="VLM圖片Caption生成器")
    parser.add_argument("--file_path", nargs='+', help="訓練資料路徑列表")
    parser.add_argument("--load_model", type=str, default="./results/clip_ft.pt", help="是否使用先前儲存的模型檔")
    parser.add_argument("--enable_data_opt", type=bool, help="是否使用 open-data 微調")
    parser.add_argument("--opt_data_exist", type=bool, help="檢查該檔案(open_data.csv)是否已存在")
    parser.add_argument("--device", type=str, choices=["cuda", "mps","cpu"], help="設備類型")
    parser.add_argument("--epoch", type=int, default=10, help="訓練回合數")
    parser.add_argument("--batch", type=int, default=16, help="訓練批量大小")
    parser.add_argument("--lr", type=int, default=1e-4, help="學習率")
    args = parser.parse_args()
    if not args.file_path:
        parser.error("請提供 --file_path 參數")
    if args.enable_data_opt:
        data_path =process_dataset(
            csv_files = args.file_path,
        )
        print("✅ Finished merge data list!")
    else:
        if not args.opt_data_exist:
            # 使用開源資料
            data_path = asyncio.run(load_opendataset())
            print("✅ Finished data preprocessing!")
        else:
            data_path = args.file_path    
    print("Using the dataset in", data_path)
    if  args.load_model:
        model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
        state_dict = torch.load(args.load_model, map_location=args.device)
        model.load_state_dict(state_dict)
        print("✅ Fine-tuned weights loaded!")
    else:
        model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
    # clip_preprocess 已包含 Resize image 為 224
    img_paths, captions = read_caption(data_path, is_deduplicated=True)
    dataset  = ImageTextDataset(img_paths, captions, clip_preprocess)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=3)
    # Define fine-tuning parameters
    epochs = args.epoch
    learning_rate = args.lr
    # Define loss function (CLIP's loss function)
    # CLIP's training objective involves aligning image and text embeddings,
    # which is handled internally when you pass both image and text through the model
    # and compute the symmetric cross-entropy loss.
    # We can define an optimizer to update the model's parameters.
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.98),
                            eps=1e-6, weight_decay=0.01)
    loss_history = training( 
        epochs=epochs, 
        model= model, 
        optimiser=optimiser,
        loader=loader,
        device=args.device
    )
    loss_plot(epochs,loss_history)
    

if __name__ == "__main__":
    exit(start_tuning()) 