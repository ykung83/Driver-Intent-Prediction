# Driver-Intent-Prediction

This is a repo for the machine learning code and trained algorithms. 

# INSTALLATION

Download the following directories to you dataset root directory
```
face_camera             - original B4C face cam data
face_camera_processed   - processed cabin gaze/pose labels
road_camera             - original B4C road data
road_camera_processed   - processed raw road object detections
```

Set the `create_dataset` optional parameter to True while initializing the B4CDataset dataloader to generate the full dataset. Then, run the following line: 
```
dataset = B4CDataset(cfg, split="train", create_dataset=False)
``` 

Your dataset root directory should contain the following when finished.
```
face_camera                     - original B4C face cam data
face_camera_processed           - processed cabin gaze/pose labels
road_camera                     - original B4C road data
road_camera_processed           - processed raw road object detections
road_camera_processed_combined  - processed raw road object detections
ImageSets_face_camera           - train, val, test splits for cabin data
ImageSets_road_camera           - train, val, test splits for road data
```

# GETTING_STARTED
TODO: fill in with how to train/eval models after processing