# ANO
```
git clone https://github.com/YihengZhang-dwbh/ANO.git
cd ANO/Traditional_RL
```

```
conda create -n ano_rl python=3.10 -y
conda activate ano_rl
pip install torch torchvision
pip install "numpy<2.0" gymnasium==0.29.1 "gymnasium[atari,accept-rom-license]" cleanrl wandb tensorboard envpool pandas
```

For the game Atlantis, we enforce a maximum episode length of 108,000 frames (27,000 steps with frame skip 4) and ensure proper score truncation handling. This is to prevent infinite gameplay loops common in this environment and to ensure valid logging of episodic returns.

