# PokemonPlatinum.AI 👾
Developing a Reinforcement Learning model to learn to play Pokemon Platinum, integrating this into the code through a desktop emulator such as DeSmuME. Will then utilise RLHF (Reinforcement Learning with Human Feedback) for model training.

# Directory 📍
PokemonModelLSTM.py: Modular script that contains the model architecture<br>
RLHF_Scripts: Folder that contains scripts and modular code for RLHF Model<br>
|—— modular_scripts: Folder for modular scripts to be used used during different RLHF Training Phase<br>
|——|–– load_model.py: Loads model, the state dict (dependent on phase), to be used in RLHF Training<br>
|——|–– rlhf_utils.py: A collection of functions to be used throughout training (connecting with emulator, taking actions, calculating rewards, etc.)<br>
|—— rlhf_phase1.py: Script for training initial model for Phase 1 goal<br>
|—— rlhf_phase2.py: Script for training Phase 1 model for Phase 2 goal<br>
|—— rlhf_phase3.py: Script for training Phase 2 model for Phase 3, final goal<br>
runs: Folder which contains the trained annotation models<br>
annotation_model.ipynb: Notebook for training the annotation model from YOLO<br>
model_pretraining.ipynb: Notebook to pretrain the model based on gameplay screenshots before RLHF<br>
requirements.txt<br>
.gitignore<br>
screenshots.py: Script to take screenshots of gameplay and store them for training<br>
video_extraction.py: Script to extract gameplay footage from videos of gameplay<br>

# Overall Project Workflow ✅
1. 
2. 

# Training Phases 🏋🏽
The overall goal of this project is to train model to learn to get from *Jubilife City* to *Oreburgh City*, beating at least one npc trainer along the way. For context, to get from *Jubilife City* to *Oreburgh City* in Pokémon Platinum, the user has to go from *Jubilife City*, pass through *Route 203*, then through *Oreburgh Cave*, then enters *Oreburgh City*.

For a human, this process should take no more than a few minutes.

**Phase 1:** Train the model to leave *Jubilife City* through *Route 203* 
**Phase 2:** Once the model has left *Jubilife City* into *Route 203*, train the model to challenge, and beat, one trainer npc
**Phase 3:** Finally, train the model to go through *Oreburgh Cave* to *Oreburgh City*

Once the model has entered *Oreburgh City*, it will have been successful and close.


