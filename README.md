

# vinvl_bert

<a href="https://colab.research.google.com/github/Mahmood-Anaam/vinvl_bert/blob/last/notebooks/inference_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Overview

**vinvl_bert** is a Python package for generating **Arabic image captions** using **Bidirectional Transformers (BiT)**. This library is designed to provide high-quality and accurate captions for Arabic datasets by leveraging pre-trained deep learning models.


## Installation

Clone the repository:

```bash
git clone https://github.com/Mahmood-Anaam/vinvl_bert.git
cd vinvl_bert
pip install -e .
```

or 

```bash
pip install git+https://github.com/Mahmood-Anaam/vinvl_bert.git
```


Create  .env for environment variables:

```env
HF_TOKEN = "hugging_face_token"
```

Create  conda environment:

```bash
conda env create -f environment.yml
conda activate sg_benchmark
```

Install Scene Graph Detection for feature extraction:

```bash
cd src\scene_graph_benchmark
python setup.py build develop
```

Download Image captioning model

```bash
cd ..
git lfs install
git clone https://huggingface.co/jontooy/AraBERT32-Flickr8k bit_image_captioning/pretrained_model
```

Install BiT-ImageCaptioning for image captioning:

```bash
cd ..
python setup.py build develop
```



## Quick Start

```python
import torch
from vinvl_bert.feature_extractors.vinvl import VinVLFeatureExtractor
from vinvl_bert.pipelines.bert_pipeline import BertImageCaptioningPipeline
from vinvl_bert.datasets.ok_vqa_dataset import OKVQADataset
from vinvl_bert.datasets.ok_vqa_dataloader import OKVQADataLoader
from vinvl_bert.modeling.bert_config import BerTConfig


# Config
cfg = BerTConfig
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.add_od_labels = True

# Extract image features
feature_extractor = VinVLFeatureExtractor(device=cfg.device,add_od_labels=cfg.add_od_labels)
# img # (file path, URL, PIL.Image, numpy array, or tensor) 
image_features = feature_extractor([img])
# return List[dict]: List of extracted features for each image.
# [{"boxes","classes","scores","img_feats","od_labels","spatial_features"},]

# Generate a caption
pipeline = BertImageCaptioningPipeline(cfg)
features,captions = pipeline([img])
print("Generated Caption:", caption)
```





