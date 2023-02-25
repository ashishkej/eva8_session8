# EVA8 Session7 Assignment

## Assignment

- Write a custom ResNet architecture for CIFAR10 that has the following architecture:
    - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    - Layer1 
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        - Add(X, R1)
    - Layer 2 
        - Conv 3x3 [256k] BN ReLu
        - MaxPooling2D
    - Layer 3
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        - Add(X, R2)
    - MaxPooling with Kernel Size 4
    - FC Layer 
    - SoftMax
- Uses One Cycle Policy such that:
    - Total Epochs = 24
    - Max at Epoch = 5
    - LRMIN = FIND
    - LRMAX = FIND
    - NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512

# Submission
[Colab Notebook](https://github.com/ashishkej/eva8_session8/blob/main/EVA8_Session8_Assignment.ipynb)

[Main Repo](https://github.com/ashishkej/eva8-pytorch-models)

- Reached Test Accuracy of ~69.6% in 24 epochs 



## References

* https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb



