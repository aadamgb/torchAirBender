# AirBender

## A differentiable quadrotor simulator build in PyTorch - Master Thesis Project (⚠️ongoing work)

PyTorch-based simulator for training neural network policies to control quadrotors. The project supports several control abstractions, training tasks and includes visualization enginge built in Taichi Lang. Training can be executed in parallel environments at the GPU for efficient data collection.



<!-- <div align="center">

![Bicopter Control Visualization](/media/pc.gif)
![Bicopter Control Visualization](/media/tt.gif)

![Bicopter Control Visualization](/media/race.gif)
![Bicopter Control Visualization](/media/race_fpv.gif)

</div> -->

<div align="center">
  <img src="media/pc.gif" width="45%">
  <img src="media/tt.gif" width="45%">
  <br>
  <img src="media/race.gif" width="45%">
  <img src="media/race_fpv.gif" width="45%">
</div>

<!-- ## Project Structure

```
├── cfg/
│   └── dynamics/bicopter.yaml        # Physics params (not used for now)
├── dynamics/
│   └── bicopter_dynamics.py          # 2D bicopter physics model
├── utils/
│   ├── nn.py                         # Neural network policy
│   ├── rand_traj_gen.py              # Harmonic trajectory generator
│   ├── renderer.py                   # Pygame visualization
├── train.py                          # Training script
├── test_model.py                     # Policy evaluation & visualization
└── outputs/                          # Saved policies 
``` -->

## Overview
### 📉Loss 

$$\mathcal{L}(\mathbf{x}_k^\text{ref}, \mathbf{x}_k) =
\lambda_p \|\mathbf{p}_k^\text{ref} - \mathbf{p}_k\|_2 +
\lambda_v \|\mathbf{v}_k^\text{ref} - \mathbf{v}_k\|_2 +
\lambda_a \left(1 - \mathbf{\hat{b}}_{3,k} \cdot \mathbf{z}_k \right) +
\lambda_\omega \|\boldsymbol{\omega}_k\|_2^2
$$

<!-- When selecting env=tt. The quadrotor's objective is to match the state of a reference trajectory -->


### 🧠Neural Network 
⚠️ TODO


<!-- #### Observation Input 

#### Action Output -->

### 🚁Bicopter Dynamics 
⚠️ TODO

### 🏋️Training

The training logic is implemented in [`train.py`](train.py). The framework supports training on both CPU and GPU with parallel environments for efficient data collection. Training runs for a configurable number of episodes, each containing a specified number of steps. The optimization horizon $T$ is user-configurable and determines when the acumulated loss $\mathcal{L}$ is backpropageted through time. 


## Usage

### Training
```bash
python train.py
```
<!-- Trains a policy for the control mode specified by the parameter [`cm`](https://github.com/aadamgb/diff-RL/blob/2d501c9cc4309554f6ee41fb654d6027d55d48af/train.py#L26). The policy will be saved in `outputs/{cm}.pt`. -->

### Evaluation & Visualization
<!-- To evaluate the policy, add it to this [dict](https://github.com/aadamgb/diff-RL/blob/2d501c9cc4309554f6ee41fb654d6027d55d48af/test_model.py#L87) then run: -->

```bash
python test_model.py
```
Multiple policies can be loaded simultaneously. 

## Requirements

- PyTorch
- Taichi Lang
- NumPy
- Matplotlib