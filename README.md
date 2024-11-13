# PokemonPlatinum.AI 👾
Developing a Reinforcement Learning model to learn to play Pokemon Platinum, integrating this into the code through a desktop emulator such as DeSmuME. Will then utilise RLHF (Reinforcement Learning with Human Feedback) for model training.

**Full project report:** https://scottpitcher.github.io/#pokemonaipage

# Project Overview ✅
1. Develop YOLOv9 computer-vision based model to annotate gameplay states
2. Build and hyperparameter tune the gameplay model (PyTorch DQN)
3. Pretain gameplay model on gameplay states, actions, and annotations
4. Continue training gameplay model with reinforcement learning with human feedback (RLHF)
5. Containerize and deploy model (Docker, FastAPI)

# RLHF Training Phases 🏋🏽
The overall goal of this project is to train model to learn to get from *Jubilife City* to *Oreburgh City*, beating at least one npc trainer along the way.
For context, to get from *Jubilife City* to *Oreburgh City* in Pokémon Platinum, the user has to go from *Jubilife City*, pass through *Route 203*, then through *Oreburgh Cave*, then enters *Oreburgh City*.

For a human, this process should take no more than a few minutes. For the AI... we'll find out!

**Phase 1:** Train the model to leave *Jubilife City* through *Route 203* 
**Phase 2:** Once the model has left *Jubilife City* into *Route 203*, train the model to challenge, and beat, one trainer npc
**Phase 3:** Finally, train the model to go through *Oreburgh Cave* to *Oreburgh City*

Once the model has entered *Oreburgh City*, it will have been successful and close

# Directory 📍
📁**Annotated_images**: Folder contained the actions, states, and labels for model pretraining before RL<br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **action_map.py:** <br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **action_prep.py:** Script that displays an image, prompts for an action, then repeats until all images have been proccessed <br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **annotations_errorcheck.py:** Script that runs through both the original and annotated filepath to ensure metadata matches<br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **file_prep.py:** Script to prep the files for usage in the models (renaming to a standardised naming system)<br>
&nbsp;&nbsp;&nbsp;&nbsp;📁 **Actions:** Folder that contains the actions (.json) for model pretraining <br>
&nbsp;&nbsp;&nbsp;&nbsp;📁 **Images:** Folder that contains the images (.png) for model pretraining<br>
&nbsp;&nbsp;&nbsp;&nbsp;📁 **Labels:** Folder that contains the labels (.txt) for model pretraining <br>
📁 **models**: Folder that contains the model architecture, and the .pth files for each phase's trained model <br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **PokemonModelLSTM.py:** Modular script that contains the model architecture<br>
📁 **RLHF_Scripts:** Folder that contains scripts and modular code for RLHF Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;📁 **human_review_logs**: Folder to hold the final state of each ep of training for human review<br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **rlhf_phase1.py:** Script for training initial model for Phase 1 goal<br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **rlhf_phase2.py:** Script for training Phase 1 model for Phase 2 goal<br>
&nbsp;&nbsp;&nbsp;&nbsp;📄 **rlhf_phase3.py:** Script for training Phase 2 model for Phase 3, final goal<br>
&nbsp;&nbsp;&nbsp;&nbsp;📁 **modular_scripts:** Folder for modular scripts to be used used during different RLHF Training Phase<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📄 **load_model.py:** Loads model, the state dict (dependent on phase), to be used in RLHF Training<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📄 **rlhf_utils.py:** Collection of functions used throughout training (emulator connection, actions, rewards, etc.)<br>
📁 **runs:** Folder which contains the trained annotation models<br>
📄 **annotation_model.ipynb:** Notebook for training the annotation model from YOLO<br>
📄 **model_pretraining.ipynb:** Notebook to pretrain the model based on gameplay screenshots before RLHF<br>
📄 **requirements.txt**<br>
📄 **screenshots.py:** Script to take screenshots of gameplay and store them for training<br>
📄 **video_extraction.py:** Script to extract gameplay footage from videos of gameplay<br>
