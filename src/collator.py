# src/collator.py
from transformers import DataCollatorForLanguageModeling

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer=None, mlm=False):
        super().__init__(mlm=mlm)
        self.tokenizer = tokenizer  # Set tokenizer to None if not provided

    def __call__(self, features):
        for f in features:
            f.pop("attention_mask", None)
        return super().__call__(features)
