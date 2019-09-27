# Neural Network Based Oyster Delineation in Drone Imagery 

This work shows the use of Mask R-CNN for detecting oyster reefs in aerial drone imagery.


## Installation

You can use the Dockerfile included in the `training/` directory of this repository to train the CNN and you can use the Dockerfile in `data_mgmt/` to manage the drone imagery and process everything. All processed data and raw data is available here: 

You will need Mask_RCNN installed in a directory at an equal level with this one. From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page download [version 2.1](https://github.com/matterport/Mask_RCNN/releases/tag/v2.1) and follow instructions here https://github.com/matterport/Mask_RCNN#installation for setting up Mask R-CNN.

## Run Jupyter notebooks
Open the `inspect_oyster_data.ipynb` or `inspect_oyster_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Oyster model

Train a new model starting from pre-trained COCO weights
```
python3 oyster.py train --dataset=/path/to/oyster/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 oyster.py train --dataset=/path/to/oyster/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 oyster.py train --dataset=/path/to/oyster/dataset --weights=imagenet
```

The code in `oyster.py` is set to train for 25K steps (100 epochs of 250 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
