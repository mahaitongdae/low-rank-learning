# Low-rank Representation Learning for Dynamical Systems

## Installation

- Install d3rlpy
  ```shell
    pip install d3rlpy hydra-core "minari[all]" wandb pillow "gymnasium[mujoco]"
  ```

- Install my custom exp_loggers
  ```shell
    git clone https://github.com/mahaitongdae/exp_logger.git
    cd exp_logger
    pip install -e .
  ```

- Other dependence
  ```shell
  pip install -e .
  ```

## Running

1. To learn dynamics from dataset
   ```shell 
   python ./scripts/main_offline_dataset_pretrain.py
   ```

2. Fit a Q-function from the pretrained representations
   ```shell 
   python ./scripts/main_offline_train_q.py
   ```

   TODO:
   - [ ] To merge this to d3rlpy implementation

3. Train a policy with the representation using offline RL algorithm
   
   ongoing. The script is in `scripts/d3rl/example.py`.

## Reference
1. The hyper parameters are managed by [hydra](https://hydra.cc/docs/intro/).

