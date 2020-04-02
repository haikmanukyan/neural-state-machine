# Neural State Machine
A pytorch implementation of the paper [Neural State Machine for Character-Scene Interactions](https://github.com/sebastianstarke/AI4Animation//blob/master/Media/SIGGRAPH_Asia_2019/Paper.pdf) by Sebastian Starke, He Zhang, Taku Komura and Jun Saito.

The paper presents a neural network based aproach for synthesizing humanoid animations from a dataset.

<div align="center">
 <img src="https://raw.githubusercontent.com/haikmanukyan/neural-state-machine/master/examples/example_net.gif" height="223px">
<img src="https://raw.githubusercontent.com/haikmanukyan/neural-state-machine/master/examples/example_data.gif" height="223px">
</div>


## Usage
To obtain the data please contact me for now. Put the data into the data directory.


To train a model run the following command from the root directory:
```
python scripts/train.py
```

To see the result of your training:
```
python scripts/test.py
```

In more detail:

### train
```
usage: train.py [-h] [--load-model LOAD_MODEL] [--comment COMMENT] [--lr LR]
                [--lr-alpha LR_ALPHA] [--batch-size BATCH_SIZE]
                [--n-epochs N_EPOCHS] [--restart-threshold RESTART_THRESHOLD]
                [--input-size INPUT_SIZE] [--n-experts N_EXPERTS]
                [--input-shape INPUT_SHAPE] [--encoders-shape ENCODERS_SHAPE]
                [--motionnet-shape MOTIONNET_SHAPE]
                [--gatingnet-shape GATINGNET_SHAPE]

Use to train a network. The network will be stored in the models directory
with an automatic name. Run from the project root. Remember to start visdom to
see the logs! Install ipdb if you want to debug your code after keyboard
interrupts

optional arguments:
  -h, --help            show this help message and exit
  --load-model LOAD_MODEL
                        Name of the model you want to load from the models
                        directory. Must be the folder name, ignoring the
                        parenthesis
  --comment COMMENT     A description to store with the trained network
  --lr LR               Initial learning rate
  --lr-alpha LR_ALPHA   Value to multiply the learning rate with after a
                        restart
  --batch-size BATCH_SIZE
                        Batch size
  --n-epochs N_EPOCHS   Number of epochs
  --restart-threshold RESTART_THRESHOLD
                        The multiplicative threshold for the gradient norm to
                        decide restarts
  --input-size INPUT_SIZE
                        Number of input dimensions
  --n-experts N_EXPERTS
                        Number of experts
  --input-shape INPUT_SHAPE
                        Dimensions of each part of the input (frame, goal,
                        environment, interaction)
  --encoders-shape ENCODERS_SHAPE
                        Dimensions of the hidden layer in each of the encoders
  --motionnet-shape MOTIONNET_SHAPE
                        The architecture of the motion prediction network
  --gatingnet-shape GATINGNET_SHAPE
                        The architecture of the gating network
```
### visualize
```
usage: visualize.py [-h] [-n N] [--starting-clip STARTING_CLIP] [--save-anim]
                    [--save-dir SAVE_DIR] [--predict-phase]
                    [--predict-trajectory] [--model-name MODEL_NAME]
                    [--phase-mult PHASE_MULT] [--show-data] [--show-phase]
                    [--show-input]

optional arguments:
  -h, --help            show this help message and exit
  -n N                  Number of clips to visualize
  --starting-clip STARTING_CLIP
                        Index of the first clip to visualize
  --save-anim           Save the animation as a gif
  --save-dir SAVE_DIR
  --predict-phase       Use the phase predicted by the network
  --predict-trajectory  Use thetrajectory predicted by the network
  --model-name MODEL_NAME
                        The name of the model you want to test, just give the
                        folder name ignoring the parenthesis
  --phase-mult PHASE_MULT
                        Multiply the speed of the phase update by this value.
                        Works only when --predict-phase is on
  --show-data           Visualize the ground truth
  --show-phase          Visualize the phase
  --show-input          Visualize the actual input to the network
```
## Reference
 
A repository by the authors can be found [here](https://github.com/sebastianstarke/AI4Animation/)