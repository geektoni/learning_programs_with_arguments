# Learning compositional programs with arguments and sampling

## Project Description

One of the most challenging goals in designing intelligent systems is empowering them with the ability to synthesize programs from data. Namely, given specific requirements in the form of input/output pairs, the goal is to train a machine learning model to discover a program that satisfies those requirements.
A recent class of methods exploits combinatorial search procedures and deep learning to learn compositional programs. However, they usually generate only toy programs using a domain-specific language that does not provide any high-level feature, such as function arguments, which reduces their applicability in real-world settings.
We extend upon a state of the art model, AlphaNPI, by learning to generate functions that can accept arguments. This improvement will enable us to move closer to real computer programs.
We showcase the potential of our approach by learning the Quicksort algorithm, showing how the ability to deal with arguments is crucial for learning and generalization.  

We use a basis the code of a previous paper by Pierrot et al. "Learning Compositional Neural Programs with Recursive Tree Search and Planning" (https://arxiv.org/abs/1905.12941).

## Directory Structure

The project is structured as follow:
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
