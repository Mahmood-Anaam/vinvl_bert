import torch

class VinVLBertConfig:
    """
    Configuration class for VinVLBertPipeline.
    This class provides all the necessary settings for initializing and running the pipeline.
    """

    # General settings
    model_id = "jontooy/AraBERT32-Flickr8k"  # model id from huggingface 
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Device for computation (GPU/CPU)

    # Dataset settings
    dataset_path = "MahmoodAnaam/vqav2-ar-en-validation-2"  # Path or name of the dataset
    language = "ar"  # Language for questions/answers ("ar" for Arabic, "en" for English)
    split = "validation"  # Dataset split to use ("train", "validation", "test")
    save_shard_dir = "./output/VQAv2_Features_Checkpoint"  # Directory to save extracted features

    num_shards = 10  # ow many shards to split the dataset into.
    current_index_shard = 0  # Which shard to select and return
    username = "MahmoodAnaam"  # Username for Hugging Face

    # Image and object detection settings
    add_od_labels = True  # Whether to add object detection labels to input
    max_img_seq_length = 50  # Maximum sequence length for image features

    # Text input settings
    max_seq_length = 70  # Maximum sequence length for text input
    max_seq_a_length = 40  # Maximum sequence length for primary text (e.g., question)
    is_train = False  # Whether the configuration is for training or inference
    mask_prob = 0.15  # Probability of masking tokens during training
    max_masked_tokens = 3  # Maximum number of tokens to mask in a single sequence

    # DataLoader settings
    batch_size = 30  # Number of samples per batch
    num_workers = 2  # Number of workers for data loading
    shuffle = False  # Whether to shuffle the dataset
    pin_memory = False  # Whether to use pinned memory (for CUDA optimization)
    drop_last = False  # Whether to drop the last incomplete batch
    seed = 42  # Random seed for reproducibility

    # Generation settings
    is_decode = True  # Enable decoding (generation mode)
    do_sample = False  # Whether to use sampling for generation
    bos_token_id = None  # Beginning of sentence token ID (will be set by tokenizer)
    pad_token_id = None  # Padding token ID (will be set by tokenizer)
    eos_token_ids = None  # End of sentence token ID(s) (will be set by tokenizer)
    mask_token_id = None  # Masking token ID (will be set by tokenizer)
    max_gen_length = 50  # Maximum length for generated text
    num_beams = 5  # Number of beams for beam search
    temperature = 1.0  # Temperature for sampling (lower values make output more deterministic)
    top_k = 50  # Top-k sampling (0 disables it)
    top_p = 1.0  # Top-p (nucleus) sampling (0 disables it)
    repetition_penalty = 1.0  # Penalty for repeating words (1.0 disables it)
    length_penalty = 1.0  # Penalty for sequence length (used in beam search)
    num_return_sequences = 1  # Number of sequences to return
    num_keep_best = 3  # Number of best sequences to keep

    # Constrained Beam Search (CBS) settings
    use_cbs = False  # Whether to use constrained beam search
    min_constraints_to_satisfy = 0  # Minimum number of constraints to satisfy (if CBS is enabled)

# ........................................................................................
