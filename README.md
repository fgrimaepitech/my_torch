my_torch is better than pytorch lol.

## Need to Implement:

### A. Essential Layers:

- [x] Batch Normalization - Critical for stable training (BatchNorm1d, BatchNorm2d)
- [x] Dropout - For regularization
- [x] Pooling layers (MaxPool2d, AvgPool2d)
- [x] Flatten layer - To convert conv outputs to linear inputs
- [x] Residual blocks - For deeper networks

### B. Activation Functions:

- [ ] Sigmoid - For binary classification
- [ ] Tanh - For value estimation (-1 to 1)
- [ ] LeakyReLU - Better than ReLU in some cases
- [ ] Softmax - For move probability distribution

### C. Loss Functions:

- [ ] MSE Loss - For value prediction
- [ ] CrossEntropy Loss - For move prediction
- [ ] Huber Loss - More robust than MSE
- [ ] Custom chess loss - Combine policy and value loss
