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
- 💾 **Retrieval demo** text query → top-K images with similarity scores and preview.

### File Structure
```text
.
├── src/
│   ├── caption_generation.py   # VLM captioning helpers & CLI
│   ├── polish_sentance.py      # Async polish/translate captions to zh-TW
│   ├── preprocess_data.py      # Dataset utils: merge, expand, deduplicate, transforms
│   ├── fine_tuning.py          # CLIP fine-tuning loop + loss plotting
│   └── main.py                 # Inference demo: text → top-K similar images
├── original_dataset/           # Your images
├── results/                    # Saved weights, plots, artifacts
└── README.md     
```

### Quick Start
1. Environment
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adapt to your CUDA
   pip install pillow numpy matplotlib tqdm pandas datasets requests
   pip install googletrans==4.0.0-rc1 ollama  # optional (captioning/polishing)
   pip install git+https://github.com/openai/CLIP.git
   ```
   | If you don’t use Ollama or googletrans, you can skip installing them and the related steps below.
---
2. (Optional) Generate Captions with a VLM
   caption_generation.py uses Ollama (e.g., qwen2.5vl:7b) to describe images in zh-TW.
   ```
   # Single image
   python src/caption_generation.py --image_path /path/to/image.jpg --output_file captions.csv
   # A directory of images
   python src/caption_generation.py --image_dir /path/to/images --output_file captions.csv --batch_size 8
   ```
   Output (captions.csv):
   | image_path	| Caption |
   |--------------------|----------|
   | /path/to/image.jpg	| Complete describtion |
   
   |Notes:
   - The script will resize images and save resized copies under ../resize_dataset/.
   - The default prompt forces Traditional Chinese and discourages hallucination.
---
3. (Optional) Polish/Translate Existing Captions
   `polish_sentance.py` reads a JSONL (e.g., CrossModal-3600 format), translates/polishes to zh-TW, and writes a CSV:
    ```
   python src/polish_sentance.py
   # By default:
   #   img_dir   = ../crossmodal-3600/images
   #   text_dir  = ../crossmodal-3600/captions.jsonl
   #   output    = open_data.csv
    ```
    You’ll get open_data.csv with two columns: image_path, Caption.
---   
4. Consolidate / Optimize Your Training CSV
   `preprocess_data.py` merges multiple CSVs (or converts a single TXT/JSONL) and:
   expands multi-caption cells into multiple (image, caption) rows,
   drops NaN captions,
   optionally de-duplicates repeated phrases.
   ```
   from src.preprocess_data import process_dataset
   # Example (inside a Python shell or notebook)
   process_dataset(
       csv_files=["captions.csv", "open_data.csv"],  # at least 2 CSVs
       output_file="captions_final.csv"
   )
   ```
   Result: captions_final.csv — clean (image, caption) pairs for fine-tuning.
---
5. Fine-Tune CLIP
   ```
     python src/fine_tuning.py \
     --file_path captions_final.csv \
     --epoch 10 \
     --batch 16 \
     --lr 1e-4 \
     --device cuda
     ...
  ```

Artifacts:
   - Weights: ./results/clip_ft_1031.pt (default path inside the script)
   - Loss curve: ./results/loss.png
   | The training loop uses cosine similarity between image/text features and an AdamW optimizer. It logs per-epoch average loss and plots the curve at the end.
---
6. Run the Retrieval Demo
   Put a few sample images in ../dataset_for_demo, then:
   ```
      python src/main.py
   ```
   You’ll see console output like:
   ```
      Using device: cuda
      Loading model successfully!
      Here's the related image sample_001.jpg, similarity score=0.988
   ```
   and a Matplotlib window displaying the top-K images.
---
### File-by-File Analysis
`caption_generation.py` — VLM Image Captioning (zh-TW)
- Purpose: Generate Traditional Chinese captions for images using a VLM (via Ollama chat).
- Key class & methods:
   - `VLMImageCaptionGenerator(model_name="qwen2.5vl:7b")`
      - `load_image(image_path)`: handles local file or URL, converts to RGB, resizes and saves a copy to `../resize_dataset/`.
      - `generate_caption(image_path, max_length=None)`: single-image captioning.
      - `generate_batch_captions(image_paths, max_length=None, num_workers=4)`: parallel captioning with ThreadPoolExecutor.
      - `save_captions_to_file(image_paths, captions, output_file)`: appends to CSV with header handling.
- CLI:
   - `--image_path` or `--image_dir`, `--output_file`, `--batch_size`, `--max_length`.
- I/O:
   - Input: path(s) to `.jpg/.png` or URLs.
   - Output: CSV file with `image_path, Caption`.
---
`polish_sentance.py` — Async Caption Polishing / Translation
 - Purpose: Provide utilities for transforms, dataset wrapping, dataframe expansion and merging.
 - Key components:
   - `build_preprocess(n_px=224)`: torchvision `Compose` matching CLIP preprocessing (resize, center crop, normalize).
   - `ImageTextDataset(image_paths, texts, transform)`: minimal dataset class for DataLoader.
   - `aggregate_image_captions(image_path, captions)`: split one cell containing multi-captions (e.g., lines 1., 2), or \n) into multiple `(image_path, caption)` pairs.
   - `expand_dataframe(df)`: expand a dataframe by applying the above split for each row.
   - `deduplicate_contxt(text, sep="，")`: remove repeated phrases separated by “，”.
   - `process_dataset(csv_files=None, txt_file=None, output_file="captions_final.csv")`:
      - If **multiple CSVs** are given, merges them, drops NaN captions, expands multi-captions, saves to CSV.
      - If a **single TXT/JSONL** is given, converts accordingly.
---
### `fine_tuning.py` — CLIP Training Loop
Purpose: Fine-tune CLIP (ViT-B/32) with your (image, caption) CSV.
   - Key functions:
      - `load_data_from_csv(raw_data_path, is_deduplicated=False, is_aggregated=False)`: read CSV, optional de-duplication, return lists of image paths and captions.
      - `build_loader(img_paths, captions, batch_size, n_px=224)`: build a PyTorch DataLoader with CLIP-style transforms.
      - `training(epochs, model, optimiser, loader, device)`: cosine similarity objective between image/text embeddings; logs epoch loss; saves weights to results/clip_ft_1031.pt.
      - `loss_plot(epochs, loss_history)`: saves a loss curve to results/loss.png.
      - `start_tuning() (CLI)`: parses args and runs the full process.
   - CLI args:
      `--file_path`: one or more CSV files (merged if multiple).
      `--enable_data_opt`, `--opt_data_exist`: optional open-data path creation via process_dataset.
      `--epoch`, `--batch`, `--lr`, `--device`.

---
### Roadmap
- Add FAISS or Annoy for scalable retrieval.
- Expose a Streamlit/Gradio UI for live demos.
- Integrate a reranker (e.g., cross-encoder) for better Top-K ordering.
- Optional YOLO-based cropping + captioning of object regions.
- Dockerfile & Colab notebook.

---
### Acknowledgements
- OpenAI CLIP (ViT-B/32)
- HuggingFace datasets (Crossmodal-3600)
- Ollama (for local VLMs)
- googletrans (fast prototype translation)

