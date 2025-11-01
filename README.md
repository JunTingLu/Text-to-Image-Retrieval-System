### 🧠 Text-to-Image Retrieval System 
Developed an OpenCLIP-based text-to-image retrieval system by integrating YOLO-based object cropping, automatic caption generation, and embedding fine-tuning, enhancing semantic alignment and retrieval accuracy. Leveraged contrastive learning to effectively learn discriminative representations from the image dataset.

<img width="500" height="300" alt="CLIP 框架示意圖" src="https://github.com/user-attachments/assets/30229dbb-e3a5-4da9-a5f2-d7cfc8fae5fd" />
<br> 此圖取自 Roboflow 的文章，展示圖文嵌入在語意空間中的對齊方式。  

### Features
- 🔍 **Implemented natural language–based text-to-image retrieval**, enabling semantic search over visual datasets with CLIP-style embeddings.
- 🖼️ **Developed image-to-image similarity search** leveraging feature embeddings for efficient visual content retrieval.
- 🎯 **Integrated fine-tuning pipeline** allowing domain-specific adaptation of pretrained CLIP models for higher retrieval accuracy.
- 📊 **Supported custom datasets with image augmentation**, leveraging modular data loaders and flexible preprocessing pipelines for scalable integration.
- ⚡ **Optimized inference performance with GPU-accelerated** embedding computation and batched similarity search.
- 💾 **Built persistent vector storage layer (FAISS)** to enable fast saving, reloading, and querying of large-scale embeddings.

### File Structure



### Installation
```
pip install -r requirements.txt
```
### Quick Start
1. Search with query
2. Run Complete Example

3. Parameters


### Datasets
1. Custom Images
- Support local image files (png/jpeg/jpg)
- Support network image URLs
- Support PIL Image objects

2. Fliker 8k


---
### 🧠 Stretagy
