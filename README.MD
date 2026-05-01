# Atari Breakout DQN & Curriculum Learning

CS/DS 541 Final Project

## Setup

1. **Install Requirements:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**

All hyperparameters for the environment, agent, and training loop are defined in `config.yaml`

3. **Running the Experiment**

- This script will sequentially train all four agents. 

- Models will be saved automatically in the `checkpoints/` directory.

- Models will be saved automatically in the `logs/` directory.

The experiment can then be run from main.py in the root directory

   ```bash
   python src/main.py
   ```

Performance can be monitored on Tensorboard

   ```bash
   tensorboard --logdir logs/
   ```


4. **Testing Agents**

Quick test: Testing a model on easy mode over 100 episodes

```bash
python src/test.py --model_path checkpoints/easy_only_final.pth --test_episodes 100 --difficulty 0
```


### Arguments List:
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model_path` | Path to the `.pth` checkpoint file | `None` |
| `--config_path` | Path to `config.yaml` file | `8080` |
| `--difficulty` | Difficulty level (0 for easy, 1 for hard) | `0` |
| `--test_episodes` | Number of episodes to run for evaluation | `100` |
| `--render` | Render the game | `false` |