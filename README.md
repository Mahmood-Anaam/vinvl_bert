# vinvl_bert

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmood-Anaam/vinvl_bert/blob/main/notebooks/vinvl_bert_demo.ipynb)

## Overview

The `vinvl_bert` repository provides a framework for generating Arabic image captions by leveraging pre-trained Bidirectional Transformers (BiT). It processes visual features from images and integrates them with textual data to produce accurate and context-aware captions. This repository is tailored for Arabic datasets and supports extensive customizations for various use cases in vision-language tasks.

## Features
- **Integration with pre-trained models**: Easily use pre-trained models for captioning tasks.
- **Image and text feature fusion**: Incorporates image region features with textual data.
- **Flexible configurations**: Supports various decoding methods and customizations for generation.
- **Supports constrained beam search (CBS)**: Enables fine-grained control over output captions.

## Installation

Install the package and its dependencies:

```bash
pip install git+https://github.com/Mahmood-Anaam/vinvl.git --quiet
pip install git+https://github.com/Mahmood-Anaam/vinvl_bert.git --quiet
```

## Quick Start

```python
import torch
from PIL import Image
import requests
from vinvl_bert.feature_extractors import VinVLFeatureExtractor
from vinvl_bert.pipelines import VinVLBertPipeline
from vinvl_bert.configs import VinVLBertConfig

# Configure settings
cfg = VinVLBertConfig()
cfg.model_id = "jontooy/AraBERT32-Flickr8k"  # model id from huggingface
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"  # Device for computation (GPU/CPU)

# Image and object detection settings
cfg.add_od_labels = True  # Whether to add object detection labels to input
cfg.max_img_seq_length = 50  # Maximum sequence length for image features

# Generation settings
cfg.is_decode = True  # Enable decoding (generation mode)
cfg.do_sample = False  # Whether to use sampling for generation
cfg.max_gen_length = 50  # Maximum length for generated text
cfg.num_beams = 5  # Number of beams for beam search
cfg.temperature = 1.0  # Temperature for sampling (lower values make output more deterministic)
cfg.top_k = 50  # Top-k sampling (0 disables it)
cfg.top_p = 1.0  # Top-p (nucleus) sampling (0 disables it)
cfg.repetition_penalty = 1.0  # Penalty for repeating words (1.0 disables it)
cfg.length_penalty = 1.0  # Penalty for sequence length (used in beam search)
cfg.num_keep_best = 3  # Number of best sequences to keep

# Load an example image
img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(img_url, stream=True).raw)

# Extract image features
feature_extractor = VinVLFeatureExtractor(device=cfg.device, add_od_labels=cfg.add_od_labels)
image_features = feature_extractor([image])

# Generate a caption
pipeline = VinVLBertPipeline(cfg)
features, captions = pipeline([image])
print("Generated Caption:", captions[0])
```

## Customization

You can fine-tune or modify configurations in `VinVLBertConfig` to suit specific tasks, such as:
- Adjusting sequence lengths for text and images.
- Modifying beam search parameters for generation.
- Enabling or disabling constrained beam search (CBS) for specific constraints.

## Limitations
This repository is a utility for integrating pre-trained models for Arabic image captioning. It is not a full-fledged library for vision-language tasks and assumes familiarity with PyTorch and Transformers.
