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
python main.py --model_type opt --seq_len 1024 --push_to_hub
python main.py --model_type mamba --seq_len 1024 --push_to_hub
```

DeepSpeed Stage3 with Multiple GPU Environment for larger sequences 
```
accelerate launch --multi_gpu main.py --model_type opt --seq_len 4096 --use_deepspeed
```
