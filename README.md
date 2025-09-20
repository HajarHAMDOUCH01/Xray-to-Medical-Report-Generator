---
# X-Ray Report Generator

This repository contains the code for the `hajar001/xray_report_generator` model in hugging face, which is designed to generate medical reports from X-ray images.

---

## How to Test This Model


**Hugging face**

go to [Hugging face repository](https://huggingface.co/hajar001/xray_report_generator)

**Local usage**

To test the model and generate reports locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HajarHAMDOUCH01/report_generator
    cd report_generator 
    ```

2.  **Download Model Weights:**
    You can download the necessary model weights from the Hugging Face repository: [LINK_TO_HUGGING_FACE_REPO](https://huggingface.co/hajar001/xray_report_generator)

3.  **Place Weights in Configuration:**
    After downloading the weights, update their paths in the `configs.constants.py` file within this repository.

4.  **Run Inference:**
    Example prediction command :
    all arguments are in predict.py
    ```bash
      python predict.py \
    --image_path xray.jpg \
    --model_path final_model.pth \
    --prompt_text "prompt text" \
    --max_length 256 \
    --num_beams 5 \
    --do_sample \
    --top_p 0.95
    ```
---


This model generates medical reports from X-ray images using a multi-modal architecture combining:

- **BiomedCLIP**: Vision encoder for medical image understanding (frozen during training)
- **Q-Former**: Cross-modal alignment between vision and language
- **BioGPT**: Medical text generation model

## Model Architecture

The model consists of three main components:
1. A fine tuned then frozen BiomedCLIP encoder that processes X-ray images and extracts 512-dimensional features
2. A Q-Former with 32 query tokens that bridges visual features to language representations  
3. A BioGPT decoder that generates medical reports
4. An optional projection layer between Q-Former (768-dim) and BioGPT (1024-dim)

## Installation Requirements

```bash
pip install --upgrade transformers
pip install torch open_clip_torch Pillow
pip install sacremoses
pip install tqdm
```
## Model Components Details

### BiomedCLIP Encoder
- **Model**: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Feature dimension**: 512
- **Status**: Frozen during training to preserve fine tuned medical vision knowledge
- **Input**: RGB X-ray images
- **Output**: L2-normalized 512-dimensional feature vectors

### Q-Former Configuration
- **Hidden size**: 768
- **Layers**: 6 transformer layers
- **Attention heads**: 12
- **Query tokens**: 32 learnable tokens
- **Cross-attention**: Enabled for vision-language alignment
- **Purpose**: Bridge between visual features (512-dim) and language model (1024-dim)

### BioGPT Language Model
- **Base model**: `microsoft/biogpt`
- **Hidden size**: 1024
- **Specialization**: Biomedical text generation
- **Max sequence length**: 256 tokens
- **Tokenizer**: BioGPT tokenizer with medical vocabulary

### Projection Layer
- **Input**: Q-Former output (768-dim)
- **Output**: BioGPT input (1024-dim)
- **Type**: Linear transformation
- **Purpose**: Dimension alignment between Q-Former and BioGPT

## Training Details

- **Training Strategy**: 
  - BiomedCLIP: Frozen (maintains pre-trained medical vision knowledge)
  - Q-Former: Fine-tuned for vision-language alignment
  - Projection layer: Trained from scratch
  - BioGPT: Fine-tuned for medical report generation

- **Training Configuration**:
  - Batch size: 4 (training), 8 (evaluation)  
  - Learning rate: 5e-5
  - Epochs: 3
  - Max sequence length: 256 tokens
  - Gradient accumulation: 1 step

- **Loss Function**: Cross-entropy loss on report tokens (query tokens ignored in loss calculation)

## Generation Parameters

The model supports various generation strategies:

- **Beam Search**: `num_beams=3` for more focused generation
- **Sampling**: `do_sample=True` with `top_k` and `top_p` for diverse outputs
- **Temperature**: 0.7 (when sampling)
- **Max tokens**: Configurable via `max_new_tokens`

## Model Inputs/Outputs

### Training Mode
- **Inputs**: 
  - `image_features`: Pre-extracted features (batch_size, 512)
  - `input_ids`: Tokenized reports (batch_size, seq_len)
  - `attention_mask`: Attention masks (batch_size, seq_len)
- **Output**: Cross-entropy loss value

### Inference Mode
- **Inputs**:
  - `image_path`: Path to X-ray image (string)
  - OR `image_features`: Pre-extracted features (batch_size, 512)
  - `prompt_text`: Optional text prompt (string)
  - Generation parameters: `max_new_tokens`, `num_beams`, etc.
- **Output**: Generated medical report (string)

## Technical Notes

1. **Device Handling**: Model automatically detects and uses CUDA if available
2. **Image Preprocessing**: Uses BiomedCLIP's preprocessing pipeline
3. **Feature Normalization**: Image features are L2-normalized
4. **Token Handling**: Automatic BOS/EOS token addition for reports
5. **Memory Efficiency**: Supports gradient checkpointing for large models

## Limitations

- Requires `open_clip_torch` for BiomedCLIP functionality
- Model size: ~1.5B parameters (BioGPT: ~1.5B, Q-Former: ~124M)
- GPU memory: Recommend 8GB+ VRAM for inference
- Image input: Currently supports single image per inference

## Citation

If you use this model, please cite the original papers:

```bibtex
@article{zhang2024biomedclip,
  title={A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs},
  author={Sheng Zhang and Yanbo Xu and Naoto Usuyama and Hanwen Xu and Jaspreet Bagga and Robert Tinn and Sam Preston and Rajesh Rao and Mu Wei and Naveen Valluri and Cliff Wong and Andrea Tupini and Yu Wang and Matt Mazzola and Swadheen Shukla and Lars Liden and Jianfeng Gao and Angela Crabtree and Brian Piening and Carlo Bifulco and Matthew P. Lungren and Tristan Naumann and Sheng Wang and Hoifung Poon},
  journal={NEJM AI},
  year={2024},
  volume={2},
  number={1},
  doi={10.1056/AIoa2400640},
  url={https://ai.nejm.org/doi/full/10.1056/AIoa2400640}
}

@article{10.1093/bib/bbac409,
    author = {Luo, Renqian and Sun, Liai and Xia, Yingce and Qin, Tao and Zhang, Sheng and Poon, Hoifung and Liu, Tie-Yan},
    title = "{BioGPT: generative pre-trained transformer for biomedical text generation and mining}",
    journal = {Briefings in Bioinformatics},
    volume = {23},
    number = {6},
    year = {2022},
    month = {09},
    abstract = "{Pre-trained language models have attracted increasing attention in the biomedical domain, inspired by their great success in the general natural language domain. Among the two main branches of pre-trained language models in the general language domain, i.e. BERT (and its variants) and GPT (and its variants), the first one has been extensively studied in the biomedical domain, such as BioBERT and PubMedBERT. While they have achieved great success on a variety of discriminative downstream biomedical tasks, the lack of generation ability constrains their application scope. In this paper, we propose BioGPT, a domain-specific generative Transformer language model pre-trained on large-scale biomedical literature. We evaluate BioGPT on six biomedical natural language processing tasks and demonstrate that our model outperforms previous models on most tasks. Especially, we get 44.98\%, 38.42\% and 40.76\% F1 score on BC5CDR, KD-DTI and DDI end-to-end relation extraction tasks, respectively, and 78.2\% accuracy on PubMedQA, creating a new record. Our case study on text generation further demonstrates the advantage of BioGPT on biomedical literature to generate fluent descriptions for biomedical terms.}",
    issn = {1477-4054},
    doi = {10.1093/bib/bbac409},
    url = {https://doi.org/10.1093/bib/bbac409},
    note = {bbac409},
    eprint = {https://academic.oup.com/bib/article-pdf/23/6/bbac409/47144271/bbac409.pdf},
}

```

## Contributing

We welcome contributions to this project! If you'd like to contribute, simply clone this repository and submit your changes via pull request.
