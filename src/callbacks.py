# src/callbacks.py
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class PushToHubCallback(TrainerCallback):
    def __init__(self, model_type, checkpointing_config, push_to_hub, save_to_hf_fn):
        self.model_type = model_type
        self.config = checkpointing_config
        self.push_to_hub = push_to_hub
        self.save_to_hf = save_to_hf_fn

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.push_to_hub:
            step = state.global_step
            checkpoint_path = f"{args.output_dir}/checkpoint-{step}"
            self.save_to_hf(self.model_type, checkpoint_path, self.config, step)
