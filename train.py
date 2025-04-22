import os
import time
import wandb
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, OPTConfig, OPTForCausalLM, AutoTokenizer
from config._config import CheckpointingConfig
from src.hf_utils import save_to_hf  # <- updated
from src.utils.utils import get_deepspeed_config
from src.collator import CustomDataCollator

def train_model(model_type="opt", seq_len=128, use_deepspeed=False, push_to_hub=True, dry_run=False):
    # Load the dataset
    dataset = load_dataset(f"babylm-seqlen/train_100M_{seq_len}_single_shuffle")
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

    # Dry run or full training
    if dry_run:
        train_dataset = dataset["train"].select(range(100))
        output_dir = f"./dryruns/{model_type}-babylm-{seq_len}"
    else:
        train_dataset = dataset["train"]
        output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}"

    os.makedirs(output_dir, exist_ok=True)
    checkpointing_config = CheckpointingConfig(run_name=f"{model_type}_babylm_{seq_len}")

    # Load tokenizer from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained("babylm-seqlen/tokenizer")

    if model_type == "opt":
        config = OPTConfig(
            vocab_size=50257,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            torch_dtype="float16",
        )
        model = OPTForCausalLM(config)
        data_collator = CustomDataCollator(mlm=False)  # No tokenizer needed here
        trainer_cls = Trainer

    wandb.init(
        project="babylm-seqlen",
        name=checkpointing_config.run_name,
        config={
            "model_type": model_type,
            "seq_len": seq_len,
            "use_deepspeed": use_deepspeed,
            "dry_run": dry_run,
            "push_to_hub": push_to_hub,
        },
        mode="disabled" if dry_run else "online",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        num_train_epochs=10,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=checkpointing_config.save_every_n_steps,
        save_total_limit=2,
        fp16=True,
        report_to="wandb",
        run_name=checkpointing_config.run_name,
        deepspeed=get_deepspeed_config() if use_deepspeed else None,
        logging_steps=10,
        disable_tqdm=False,
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,  # Pass the tokenizer
        data_collator=data_collator,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)  # Save tokenizer to the output directory

    if push_to_hub:
        repo_id = f"babylm-seqlen/{model_type}-babylm-{seq_len}"
        save_to_hf(model_type, output_dir, checkpointing_config, trainer.state.global_step)

    print(f"✅ Training {model_type.upper()} for seq_len {seq_len} done in {end_time - start_time:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="opt", choices=["opt", "mamba"])
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--no_push_to_hub", action="store_true", help="If set, do NOT push to the Hugging Face Hub.")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        use_deepspeed=args.use_deepspeed,
        push_to_hub=not args.no_push_to_hub,
        dry_run=args.dry_run
    )
