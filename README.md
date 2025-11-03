### 🧠 Text-to-Image Retrieval System 
Developed an OpenCLIP-based text-to-image retrieval system by integrating YOLO-based object cropping, automatic caption generation, and embedding fine-tuning, enhancing semantic alignment and retrieval accuracy. Leveraged contrastive learning to effectively learn discriminative representations from the image dataset.

<img width="500" height="300" alt="CLIP 框架示意圖" src="https://github.com/user-attachments/assets/30229dbb-e3a5-4da9-a5f2-d7cfc8fae5fd" />
<br> This image is taken from a Roboflow article and illustrates how text and images are aligned in semantic space  

### Features
- 🔍 **Implemented natural language–based text-to-image retrieval**, enabling semantic search over visual datasets with CLIP-style embeddings.
- 🖼️ **Developed image-to-image similarity search** leveraging feature embeddings for efficient visual content retrieval.
- 🎯 **Integrated fine-tuning pipeline** allowing domain-specific adaptation of pretrained CLIP models for higher retrieval accuracy.
- 📊 **Supported custom datasets with image augmentation**, leveraging modular data loaders and flexible preprocessing pipelines for scalable integration.
- ⚡ **Optimized inference performance with GPU-accelerated** embedding computation and batched similarity search.
- 💾 **Built persistent vector storage layer (FAISS)** to enable fast saving, reloading, and querying of large-scale embeddings.

### File Structure
```text
project_root/
├── original_dataset/         # Custom datasets (CSV)
├── src/                      # Source code directory
│   ├── caption_generation.py # Generate captions for images
│   ├── fine_tuning.py        # Fine-tune models
│   ├── main.py               # Main entry point of the program
│   ├── polish_sentence.py    # Polish or refine generated sentences
│   └── preprocess_data.py    # Preprocess and clean input data
└── README.md                 # Project documentation
```

### Installation
```
pip install -r requirements.txt
```
### Quick Start
1. Search with query (main.py)
   ```
  folder = "../dataset_for_demo" 
  sample_imgs = glob.glob(os.path.join(folder, "*.[jp][pn]g"))  
  sample_imgs = [i for i in sample_imgs]
  sample_feats  = encode_images(sample_imgs)
  query = "A man was smiling"
  search_results = search(query, sample_imgs, sample_feats, topk=5)
  for p, s in search_results:
      print(f"Here's the related image {p.split('/')[-1]:30s}, similarity score={s:.3f}")   
   ```

2. Run Complete Example
    ```
      

    ```
   
3. Parameters



### Datasets
1. Custom Images
- Support local image files (png/jpeg/jpg)
- Support network image URLs
- Support PIL Image objects

2. Crossmodal-3600


---
### 🧠 Stretagy
