from transformers import DataCollatorForLanguageModeling

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer=None, mlm=False):
        # Pass tokenizer as None or provide a tokenizer if needed for padding
        super().__init__(mlm=mlm)
        self.tokenizer = tokenizer  # Set tokenizer to None if not provided

    def __call__(self, features):
        # If the tokenizer is provided, we might want to ensure padding is done
        if self.tokenizer:
            return super().__call__(features)
        
        # Handle the case where no tokenizer is provided (pre-tokenized data)
        for f in features:
            f.pop("attention_mask", None)
        return features
