# NCTR: Neighborhood Consensus Transformer for feature matching
Repository of NCTR

Due to the space limitation of the paper, we post the appendix and some additional experiment results here.

The code will be released soon.

## Data preparation
### R1M dataset
1) Download R1M dataset from official [link](http://ptak.felk.cvut.cz/revisitop/revisitop1m/).
2) To split train/val/test dataset, run
    ```
    python train_val_test_r1m.py --input_dir xxx -- output_dir xxx
    ```

    This produces a validation set and a test set of 500 images, and the rest of the images are the training set.
3) Then generate random homography for validation and test sets for evaluation
    ```
    python get_perspective.py --image_dir xxx
    ```

## Qualitative Examples

### Homography Estimation
![homography](imgs/homo_compare.png)

### Outdoor Pose Estimation
![pose](imgs/pose_compare.png)

## Quantitative Results

### Homography Estimation
![homography](imgs/homo.png)

### Outdoor Pose Estimation
![pose](imgs/pose.png)

