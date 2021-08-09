## SimCLR-Pathology
Implementation of SimCLR based on [sthalles/SimCLR](https://github.com/sthalles/SimCLR), customized for training on high-resolution digital pathology images. 

## Usage
The main training script is `train.py`. Please use `python train.py --help` to see complete set of training parameters and their descriptions.

Example execution: `python train.py -out /path/to/output -data /path/to/svs/tiles  -library /path/to/csv/file.csv -j 10 --subsample .5`

The tile library (csv) should have the following format:

SlideID | x | y 
------------ | ------------- | -------------
1 | 24521 | 23426 
1 | 6732 | 3323 
1 | 1232 | 5551 
... | ... | ... 
35 | 34265 | 122 
... | ... | ... 
n | 2264 | 2436

In short, this tile library should be a record of all tile coordinates with associated slide level information (duration, event, training split, slide name).

The dataloader will load all `.svs` images located at `args.slide_path` during initiation, and pull tiles on-the-fly using the `(x,y)` coordinates during training.

The following will be generated in the output folder:
* convergence.csv
  * a file containing training loss, training concordance index, and validation condorance index over training epochs
* /clustering_grid_top
  * a folder where a clustering visualization for top 20 tiles of each cluster is displayed and saved as a `.png` 
