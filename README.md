# Deep Learning & Reinforcement Learning for Chinese Official Mahjong on Botzone

## Introduction to File Structure
- The deep learning component includes the file `trainning_data.py`, which is designed to transform initial training data into the same input-output format as Botzone, facilitating the training of DL models. It also allows for parameter adjustments to create bots capable of running on Botzone.
The files `DL_CNN_3heads_agent.py`, `DL_CNN_3heads_agent_bak.py`, `DL_MLP_3heads_agent.py`, `DL_ResNet_3heads_agent.py`, `DL_RNN_3heads_agent.py`, `DL_TIT_1head_agent.py`, `DL_TIT_3heads_agent.py`, and `DL_TIT_3heads_agent_turn.py`, `DL_VIT1_agent.py` are for comparative training of bots.

- The reinforcement learning component is comprised of the files `PolicyGradient_1head.py`, `PolicyGradient_3heads.py`, `AC_vanilla.py`, `PolicyGradient_vanilla.py`, `PPO_vanilla.py`, and `PPO_3heads.py`, each representing three distinct implementation methods.

- The data files include raw human contest data, deep learning training data, and trained models in three parts.
  + File link: [Baidu Netdisk](https://pan.baidu.com/s/1wpPBHq3MRngMQx9EAS6-aw)
  + Extraction code: agmm

## Instructions for Code Execution
**All codes are command-line executable, using `python xxx.py` will run the code with default parameters. For detailed parameter explanations, please refer to the descriptions within the specific files.**
