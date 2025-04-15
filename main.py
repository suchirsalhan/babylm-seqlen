import argparse
from train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["opt", "mamba"], default="opt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        use_deepspeed=args.use_deepspeed,
        push_to_hub=args.push_to_hub,
    )
