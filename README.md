# babylm-seqlen
How Long Can You Go? 

```
git clone https://github.com/suchirsalhan/babylm-seqlen
python3 -m venv venvs/demo; source venvs/demo/bin/activate
bash setup.sh
pip install -e .  # From the repo root with the pyproject.toml
```

Train and push models to HuggingFace Hub
```
python train.py --model_type opt --seq_len 1024 --push_to_hub 
python train.py --model_type mamba --seq_len 1024 --push_to_hub
```
You can add  `--dry_run`

Bubbles (do we need to include main.py?)

```
sh launch_torchrun.sh train.py --model_type opt --seq_len 1024 --push_to_hub
sh launch_torchrun.sh train.py --model_type mamba --seq_len 1024 --push_to_hub
```

DeepSpeed Stage3 with Multiple GPU Environment for larger sequences 
```
accelerate sbatch launch_slurm.wilkes3  --multi_gpu train.py --model_type opt --seq_len 4096 --use_deepspeed
```
