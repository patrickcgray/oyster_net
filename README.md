# Deep learning for coastal resource conservation: automating detection of shellfish reefs

This repository has the code needed to go along with the Remote Sensing in Ecology and Conservation paper "Deep learning for coastal resource conservation: automating detection of shellfish reefs." This work uses Mask R-CNN for detecting oyster reefs in aerial drone imagery.

All data, both preprocessed and original raw data, necessary to reproduce this work is available on the [Duke Digital Repository](https://research.repository.duke.edu/).


## Installation

You can use the Dockerfile included in the `training/` directory of this repository to train the CNN and you can use the Dockerfile in `data_mgmt/` to manage the drone imagery and process everything. The training Docker container is created and run using this series of commands:
```
docker build -t training_img .
docker run --name training_container --runtime=nvidia -it -p 8888:8888 -p 6006:6006 -v ~/:/host training_img
```
Once this container is running you cans start the jupyter notebook running inside of the container with:
```
jupyter notebook --allow-root --ip 0.0.0.0 /host
```
The data_mgmt container is similar except without the nvidia runtime
```
docker build -t data_mgmt_img .
docker run --name data_mgmt_container -it -p 8888:8888 -p 6006:6006 -v ~/:/host data_mgmt_img
```

You will need Mask_RCNN installed in a directory at an equal level with this one. From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page download [version 2.1](https://github.com/matterport/Mask_RCNN/releases/tag/v2.1) and follow instructions here https://github.com/matterport/Mask_RCNN#installation for setting up Mask R-CNN.

Directory structure should be:
```
data_mgmt/
training/
data/
    mosaics/
    models/
    shapefiles/
    1kx1k_dataset/  
    2kx2k_dataset/
    4kx4k_dataset/
logs/
```

## Prepare Data
Convert the original drone mosaics into tiles appropriate for deep learning analysis through this notebook: `data_mgmt/img_processing.ipynb`and shuffle and organize these tiles into training, validation, and testing datasets using `data_mgmt/training_data_management.ipynb`.

## Inspect Data

Open the `inspect_oyster_data.ipynb` jupyter notebook to explore this prepared dataset.

## Train the Oyster model

NOTE: pretrained models are available for all image sizes at: https://github.com/patrickcgray/oyster_net/releases.

Train a new model starting from pre-trained COCO weights
```
python3 oyster.py train --dataset=/path/to/oyster/dataset --weights=coco
```

Train a model from the weights used in this paper.
```
python3 oyster.py train --dataset=/path/to/oyster/dataset --weights=/path/to/weights
```

Resume training a model that you had trained earlier
```
python3 oyster.py train --dataset=/path/to/oyster/dataset --weights=last
```

The code in `oyster.py` is set to train for 25K steps (100 epochs of 250 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.

## Run the Model and Analyze Output

The model can be run at `inspect_oyster_model.ipynb` and users can both step through the detection pipelie or run it on a bulk set of images. The geolocation information for each tile output from `data_mgmt/img_processing.ipynb` can be used along with the detections to convert the CNN output back to a geolocated polygon such as a shapefile or geojson polygon.
