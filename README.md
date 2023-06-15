# Create an MNIST network from scratch

We will show you how to create a good network for mnist and train it to reach more than 99.4% test accuracy

## Initial Setup

The idea is to set up a basic end-to-end running code with augmentation, data loader, and visualization.

### Pipeline 
1. Data Loader abstracted to use separate augmentation file
2. Model moved to a separate file
3. Add Data Visualization module for Dataset and Incorrect Prediction

### Targets
1. Create a basic model which trains on MNIST Data
2. Create essential transformation using Albementation
3. Analyze Outputs

### Results
1. Prepared a network with **16,530** params
2. Trained for 15 Epochs and reached an accuracy of **99.91%** Train and **99.15%** Test

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
            Conv2d-2            [-1, 8, 24, 24]             584
         MaxPool2d-3            [-1, 8, 12, 12]               0
            Conv2d-4           [-1, 16, 12, 12]             144
            Conv2d-5           [-1, 16, 10, 10]           2,320
            Conv2d-6             [-1, 16, 8, 8]           2,320
         MaxPool2d-7             [-1, 16, 4, 4]               0
            Conv2d-8             [-1, 32, 4, 4]             544
            Conv2d-9             [-1, 32, 2, 2]           9,248
           Conv2d-10             [-1, 10, 1, 1]           1,290
================================================================
Total params: 16,530
Trainable params: 16,530
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.13
Params size (MB): 0.06
Estimated Total Size (MB): 0.20
----------------------------------------------------------------
```

```
Using Device: cuda
Epochs: 15
Lr: 0.01
Max Lr: 0.01
Batch Size: 32
Dropout: True


| Epoch | LR       | Time    | TrainLoss | TrainCorrect | TrainAcc | ValLoss  | ValCorrect | ValAcc |
|     1 | 0.010000 | 00m 11s | 0.578923  |        48272 | 80.45  % | 0.128561 |       9623 | 96.23% |
|     2 | 0.002534 | 00m 10s | 0.097078  |        58204 | 97.01  % | 0.061427 |       9796 | 97.96% |
|     3 | 0.004667 | 00m 10s | 0.069356  |        58728 | 97.88  % | 0.059053 |       9810 | 98.1 % |
|     4 | 0.006801 | 00m 11s | 0.055934  |        58936 | 98.23  % | 0.044168 |       9852 | 98.52% |
|     5 | 0.008934 | 00m 11s | 0.047059  |        59110 | 98.52  % | 0.042165 |       9866 | 98.66% |
|     6 | 0.009523 | 00m 10s | 0.039738  |        59247 | 98.75  % | 0.041785 |       9868 | 98.68% |
|     7 | 0.008571 | 00m 11s | 0.033531  |        59365 | 98.94  % | 0.032883 |       9898 | 98.98% |
|     8 | 0.007619 | 00m 11s | 0.028518  |        59476 | 99.13  % | 0.039894 |       9880 | 98.8 % |
|     9 | 0.006666 | 00m 11s | 0.024927  |        59520 | 99.2   % | 0.028359 |       9912 | 99.12% |
|    10 | 0.005714 | 00m 11s | 0.02081   |        59598 | 99.33  % | 0.034452 |       9894 | 98.94% |
|    11 | 0.004761 | 00m 11s | 0.017866  |        59649 | 99.42  % | 0.031186 |       9909 | 99.09% |
|    12 | 0.003809 | 00m 10s | 0.014664  |        59711 | 99.52  % | 0.033467 |       9900 | 99.0 % |
|    13 | 0.002857 | 00m 11s | 0.010723  |        59788 | 99.65  % | 0.030915 |       9920 | 99.2 % |
|    14 | 0.001904 | 00m 11s | 0.006672  |        59878 | 99.8   % | 0.033935 |       9913 | 99.13% |
|    15 | 0.000952 | 00m 10s | 0.003479  |        59943 | 99.91  % | 0.032759 |       9915 | 99.15% |
```

### Analysis
1. Network has more capacity to learn
2. Network is too extensive we have to optimize it to a smaller one
3. We can clearly see the network is Overfitting

## Intermediate Setup

The idea is to optimize the network further to get a robust network

### Pipeline 
1. Data Loader abstracted to use separate augmentation file
2. Model moved to a separate file
3. Add Data Visualization module for Dataset and Incorrect Prediction

### Targets
1. Create an optimized model below **8,000** params
2. Analyze Outputs

### Results
1. Prepared a network with **3,762** params
2. Trained for 15 Epochs and reached an accuracy of **98.89%** Train and **99.19%** Test

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
       BatchNorm2d-2            [-1, 4, 28, 28]               8
            Conv2d-3            [-1, 4, 28, 28]             148
       BatchNorm2d-4            [-1, 4, 28, 28]               8
            Conv2d-5            [-1, 4, 28, 28]             148
       BatchNorm2d-6            [-1, 4, 28, 28]               8
         MaxPool2d-7            [-1, 4, 14, 14]               0
           Dropout-8            [-1, 4, 14, 14]               0
            Conv2d-9            [-1, 8, 14, 14]             296
      BatchNorm2d-10            [-1, 8, 14, 14]              16
           Conv2d-11            [-1, 8, 14, 14]             584
      BatchNorm2d-12            [-1, 8, 14, 14]              16
           Conv2d-13            [-1, 8, 14, 14]             584
      BatchNorm2d-14            [-1, 8, 14, 14]              16
        MaxPool2d-15              [-1, 8, 7, 7]               0
          Dropout-16              [-1, 8, 7, 7]               0
           Conv2d-17              [-1, 8, 7, 7]             584
      BatchNorm2d-18              [-1, 8, 7, 7]              16
           Conv2d-19              [-1, 8, 7, 7]             584
      BatchNorm2d-20              [-1, 8, 7, 7]              16
           Conv2d-21              [-1, 8, 7, 7]             584
      BatchNorm2d-22              [-1, 8, 7, 7]              16
AdaptiveAvgPool2d-23              [-1, 8, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]              90
================================================================
Total params: 3,762
Trainable params: 3,762
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.25
Params size (MB): 0.01
Estimated Total Size (MB): 0.27
----------------------------------------------------------------
```
```
Using Device: cuda
Epochs: 15
Lr: 0.01
Max Lr: 0.01
Batch Size: 32
Dropout: True


| Epoch | LR       | Time    | TrainLoss | TrainCorrect | TrainAcc | ValLoss  | ValCorrect | ValAcc |
|     1 | 0.010000 | 00m 11s | 0.578923  |        48272 | 80.45  % | 0.128561 |       9623 | 96.23% |
|     2 | 0.002534 | 00m 10s | 0.097078  |        58204 | 97.01  % | 0.061427 |       9796 | 97.96% |
|     3 | 0.004667 | 00m 10s | 0.069356  |        58728 | 97.88  % | 0.059053 |       9810 | 98.1 % |
|     4 | 0.006801 | 00m 11s | 0.055934  |        58936 | 98.23  % | 0.044168 |       9852 | 98.52% |
|     5 | 0.008934 | 00m 11s | 0.047059  |        59110 | 98.52  % | 0.042165 |       9866 | 98.66% |
|     6 | 0.009523 | 00m 10s | 0.039738  |        59247 | 98.75  % | 0.041785 |       9868 | 98.68% |
|     7 | 0.008571 | 00m 11s | 0.033531  |        59365 | 98.94  % | 0.032883 |       9898 | 98.98% |
|     8 | 0.007619 | 00m 11s | 0.028518  |        59476 | 99.13  % | 0.039894 |       9880 | 98.8 % |
|     9 | 0.006666 | 00m 11s | 0.024927  |        59520 | 99.2   % | 0.028359 |       9912 | 99.12% |
|    10 | 0.005714 | 00m 11s | 0.02081   |        59598 | 99.33  % | 0.034452 |       9894 | 98.94% |
|    11 | 0.004761 | 00m 11s | 0.017866  |        59649 | 99.42  % | 0.031186 |       9909 | 99.09% |
|    12 | 0.003809 | 00m 10s | 0.014664  |        59711 | 99.52  % | 0.033467 |       9900 | 99.0 % |
|    13 | 0.002857 | 00m 11s | 0.010723  |        59788 | 99.65  % | 0.030915 |       9920 | 99.2 % |
|    14 | 0.001904 | 00m 11s | 0.006672  |        59878 | 99.8   % | 0.033935 |       9913 | 99.13% |
|    15 | 0.000952 | 00m 10s | 0.003479  |        59943 | 99.91  % | 0.032759 |       9915 | 99.15% |
```

### Analysis
1. While looking at training accuracy we can see there is a huge potential in the network to further train
2. This is a good network and can be used in further finetuning
3. While looking at failed predictions we see some of the predictions are obvious and can be trained into the network

## Final Setup

The idea is to reach a target ie 99.4% test accuracy in 15 epochs

### Pipeline 
1. Data Loader abstracted to use separate augmentation file
2. Model moved to a separate file
3. Add Data Visualization module for Dataset and Incorrect Prediction

### Targets
1. Further improve accuracy to reach above **99.4%** in 15 epochs
2. Use Custom OneCycleLR to train model
3. Use Augmentation to add Regularization to Network

### Results
1. Retrained network with **3,762** params
2. Trained for 15 Epochs and reached an accuracy of **98.94%** Train and **99.43%** Test

```
Using Device: cuda
Epochs: 15
Lr: 0.05
Max Lr: 0.1
Batch Size: 32
Dropout: True


| Epoch | LR       | Time    | TrainLoss | TrainCorrect | TrainAcc | ValLoss  | ValCorrect | ValAcc |
|     1 | 0.050000 | 00m 14s | 0.347041  |        53292 | 88.82  % | 0.075873 |       9760 | 97.6 % |
|     2 | 0.075000 | 00m 16s | 0.133324  |        57619 | 96.03  % | 0.066363 |       9797 | 97.97% |
|     3 | 0.100000 | 00m 15s | 0.107717  |        58053 | 96.75  % | 0.045474 |       9849 | 98.49% |
|     4 | 0.085000 | 00m 16s | 0.076225  |        58673 | 97.79  % | 0.042818 |       9858 | 98.58% |
|     5 | 0.070000 | 00m 17s | 0.06339   |        58873 | 98.12  % | 0.032771 |       9898 | 98.98% |
|     6 | 0.055000 | 00m 16s | 0.052222  |        59032 | 98.39  % | 0.027366 |       9909 | 99.09% |
|     7 | 0.040000 | 00m 14s | 0.049818  |        59070 | 98.45  % | 0.022665 |       9935 | 99.35% |
|     8 | 0.025000 | 00m 15s | 0.042174  |        59207 | 98.68  % | 0.021440 |       9936 | 99.36% |
|     9 | 0.010000 | 00m 17s | 0.038689  |        59291 | 98.82  % | 0.019026 |       9935 | 99.35% |
|    10 | 0.008750 | 00m 16s | 0.036276  |        59330 | 98.88  % | 0.018346 |       9937 | 99.37% |
|    11 | 0.007500 | 00m 17s | 0.0348    |        59355 | 98.92  % | 0.018045 |       9943 | 99.43% |
|    12 | 0.006250 | 00m 15s | 0.037173  |        59324 | 98.87  % | 0.018218 |       9941 | 99.41% |
```

### Analysis
1. We can clearly see the effect of using Transformation and OneCycleLR
2. We can still see there are some images that are getting miss classified these are the ones that even humans can not distinguish
3. We can still train our network further with a better training schedule