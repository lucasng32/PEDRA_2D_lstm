# PEDRA_2D_LSTM
 2D SIMULATION adapted from original PEDRA repository (https://github.com/aqeelanwar/PEDRA).
This project does not connect to AIRSIM, instead does the simulation and training in 2D.
Currently, this project is a variant of the novel FCQN (fully convolutional Q-network) architecture
described in 
 
"Deep Reinforcement Learning based Automatic
 Exploration for Navigation in Unknown
 Environment"
 by Haoran Li, Qichao Zhang, Dongbin Zhao Senior Member, IEEE (https://arxiv.org/pdf/2007.11808.pdf)

Bayesian Hilbert Maps taken from (https://github.com/RansML/Bayesian_Hilbert_Maps) <br/>
Adapted from Donovan Loh (https://github.com/CaptainKeqing/PEDRA_2D)

This project adds recurrency to the FCQN by adding a recurrent Long Short Term Memory (LSTM) layer 
after advantage and state values. Adding recurrency to DQNs (deep Q-networks) has been shown to 
be beneficial, as described in 

"Deep Recurrent Q-Learning for Partially Observable MDPs"
 by Hausknecht, Matthew and Stone, Peter (https://arxiv.org/abs/1507.06527)

# Installing
It is advisable to create a new virtual environment and install the necessary dependencies.
```
cd PEDRA_2D_lstm
pip install –r requirements.txt
```

# Demonstration
Change the directories in infer_2D to your own and run with the latest weight given in the results/weights directory.

# Training
Change the directories in main_2D to your own and run. 

# Slides and Video
Explanatory slides and 2 demo videos are included. 