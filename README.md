

# vinvl_bert

<a href="https://colab.research.google.com/github/Mahmood-Anaam/vinvl_bert/blob/main/notebooks/vinvl_bert_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Overview

**vinvl_bert** is a Python package for generating **Arabic image captions** using **Bidirectional Transformers (BiT)**. This library is designed to provide high-quality and accurate captions for Arabic datasets by leveraging pre-trained deep learning models.


## Installation

```bash
package_name = "vinvl-0.1.0-cp310-cp310-linux_x86_64.whl"
!pip install https://github.com/Mahmood-Anaam/vinvl/raw/main/{package_name} --quiet
!pip install git+https://github.com/Mahmood-Anaam/vinvl_bert.git --quiet
```

## Quick Start

```python
import torch
from PIL import Image
import requests
from vinvl_bert.feature_extractors import VinVLFeatureExtractor
from vinvl_bert.pipelines import VinVLBertPipeline
from vinvl_bert.configs import VinVLBertConfig


# Config
cfg = VinVLBertConfig
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



# e.g Image
img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(img_url, stream=True).raw)

# Extract image features
feature_extractor = VinVLFeatureExtractor(device=cfg.device,add_od_labels=cfg.add_od_labels)
# image # (file path, URL, PIL.Image, numpy array, or tensor) 
image_features = feature_extractor([image])

# return List[dict]: List of extracted features for each image.
# [{"boxes","classes","scores","img_feats","od_labels","spatial_features"},]

# Generate a caption
pipeline = VinVLBertPipeline(cfg)
features,captions = pipeline([image])
print("Generated Caption:", caption)
```





