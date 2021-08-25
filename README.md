# Learning compositional programs with arguments and sampling

**Note: this code is not production ready and things might not work straight out
of the box or they lack documentation. If you find any problems, feel free to open an issue here.**

## Project Description

Giovanni De Toni ([giovanni.det@unitn.it](giovanni.detoni@unitn.it)),
Luca Erculiani ([luca.erculiani@unitn.it](luca.erculiani@unitn.it)),
Andrea Passerini ([andrea.passerini@unitn.it](andrea.passerini@unitn.it))

We propose an architecture to perform neural synthesis of complex procedures by using deep reinforcement
learning and execution traces. The execution traces are "recipes" which specify the sequence of operations needed
to perform a certain algorithm. These traces are discovered automatically by using an Approximate Monte Carlo Tree Search (A-MCTS). Then, the network learns from these traces to execute a given algorithm. We use a deep reinforcement learning training procedure with sparse rewards.  

We use a basis a previous paper by Pierrot et al. "Learning Compositional Neural Programs with Recursive Tree Search and Planning" (https://arxiv.org/abs/1905.12941).

In our work, we try to extend this concept to learn the "function-arguments" tuple rather just which function needs to be called next. This way we will have something which resembles more closely a real computer program. To test our approach we focused mostly on sorting algorithms, namely the QuickSort procedure.  

## Directory Structure

The project is structured as follow:
* **cluster**: it contains several scripts to launch multiple experiments on a HPC cluster using the `qsub` command.
* **core**: it contains core classes which shows the underlying working principles. For instance, we have the implementation
of the MCTS procedure, the `Trainer` class used during training, etc.
* **environments**: it contains the implementation of the various environments used for training;
* **trainings**: it contains the various scripts used to train the models;
* **validation**: it contains the scripts used to validate the model and get the results;
* **visualization**: it contains the scripts which can be used to inspect the Monte Carlo Tree to extract the working execution traces. They are useful to check which is the procedure learnt by the network.
* **model**: it contains the trained model used during the experiments.


## Installation

The code was tested on **Ubuntu 18.04** by using **Python 3.7.5**. We suggest to install all the required components
by using a virtual env (e.g., conda). In order to install all the requirements, please issue this command:
```bash
cd learning_programs_with_arguments
pip install -r requirements.txt
```

## Run the experiments

The scripts to run the experiments provide many options and configurations. Please have a look at the code to check
for all the possible configurations. Here below we present the command needed to train GeneralizedAlphaNPI on the
QuickSort task

```bash
cd learning_programs_with_arguments/trainings
export PYTHONPATH=../
python train_quicksorting.py --verbose --save-model --save-results --tensorboard --widening --seed 42
```
