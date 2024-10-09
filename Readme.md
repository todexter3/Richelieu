# Richelieu
This code contains a custom implementation of a Gym environment for the Diplomacy game, and training using self-playing.

### Installation
```
# Apt installs
apt-get install -y wget bzip2 ca-certificates curl git build-essential clang-format-8 git wget cmake build-essential autoconf libtool pkg-config libgoogle-glog-dev

# Install conda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b

# Install pytorch, pybind11
conda install --yes pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
conda install --yes pybind11

# Install the required dependencies
pip install diplomacy
pip install gymnasium
pip install "stable-baselines3[extra]>=2.0.0a4"

# Make
make

```

### Project Structure
The notebook contains the following main sections:
1. Imports and Necessary Installations: Installs and imports required libraries.
2. Custom Diplomacy Class: Defines the custom environment for the Diplomacy game.
3. Training with self-playing: Trains Richelieu using self-playing.

### Custom Diplomacy Class
This section defines the `DiplomacyStrategyEnv` class, a custom implementation of a environment for the Diplomacy game. 
Key Methods
-	__init__(): Initializes the environment.
-	reset(): Resets the environment to the initial state.
-	step(action): Executes an action and returns the next state, reward, done flag, and info.
-	_init_observation_space(): Create the MultiDiscrete space for the observation space.

### Training with self-playing
This section demonstrates how to train Richelieu using self-palying. Richelieu is trained to play the Diplomacy game within the custom environment.
- train_selfplay(): training with self-playing.

Richelieu can also be trained by playing with other models.
- train(model): train by playing with other models.

### Evaluation
In this section, we evaluate Richelieu by comparing its game results with Cicero.
-	play_with(model): have Richelieu and Cicero play a game.
  
You can modified the model, removed the negotiation module, and performed massively play on no-press setting with the other six models.
It is also possible to obtain an ablation study by deleting each section and playing games with Cicero. 

### Contributing
Contributions are welcome! 

