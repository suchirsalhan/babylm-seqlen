import torch
from torch.nn import functional as F
from transformers import DataCollatorForLanguageModeling

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, mlm=False):
        super().__init__(mlm=mlm)

    def __call__(self, features):
        # We don't need to handle 'attention_mask' for pre-tokenized datasets
        for f in features:
            f.pop("attention_mask", None)  # Remove 'attention_mask' if present
        
        return super().__call__(features)
