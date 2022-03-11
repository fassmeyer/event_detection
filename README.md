# event_detection

# DETECTING ON-BALL EVENTS IN SOCCER USING INTERACTIVE RECURRENT NEURAL NETWORKS

## Preprocessing
The repo contains the presented model architecture. Before executing the training scripts, make sure to have preprocessed the datasets as described in the paper and stored as ```.npy``` files. Term the files ```trajectories.npy``` (for the trajectory data) and ```goals.npy``` (for the labels) separately for training and validation, and move them into the associated directories in ```/datasets/all``` (when using all agents). 



## Model
### Training 
To produce the best reported results, run
```
python train.py --players all --graph_model gat --adjacency_type 2 --top_k_neigh 5 --run <expname>
```
