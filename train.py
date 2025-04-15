import os, time
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import OPTForCausalLM
from config._config import CheckpointingConfig
from hf_utils import save_to_hf
from utils import get_deepspeed_config, create_output_dir
from mamba_utils import MambaLMHeadModel, MambaConfig, MambaTrainer, CustomDataCollator, save_mamba_model

def train_model(model_type="opt", seq_len=128, use_deepspeed=False, push_to_hub=False):
    dataset = load_dataset(f"babylm-seqlen/train_100M_{seq_len}_single_shuffle")
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]})
    train_dataset = dataset["train"].select(range(1000))  # Small test subset

    output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}"
    os.makedirs(output_dir, exist_ok=True)

    checkpointing_config = CheckpointingConfig(run_name=f"{model_type}_babylm_{seq_len}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    if model_type == "opt":
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
        data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
        trainer_cls = Trainer
    elif model_type == "mamba":
        config = MambaConfig(d_model=256, n_layer=6, vocab_size=50257)
        model = MambaLMHeadModel(config)
        data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
        trainer_cls = MambaTrainer

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        num_train_epochs=10,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=checkpointing_config.save_every_n_steps,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        run_name=checkpointing_config.run_name,
        deepspeed=get_deepspeed_config() if use_deepspeed else None,
        logging_steps=10,
        disable_tqdm=False,
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    if model_type == "mamba":
        save_mamba_model(model, model.config, output_dir, tokenizer)
    else:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    if push_to_hub:
        save_to_hf(model_type, output_dir, checkpointing_config)

    print(f"✅ Training {model_type.upper()} for seq_len {seq_len} done in {end_time - start_time:.2f}s")
