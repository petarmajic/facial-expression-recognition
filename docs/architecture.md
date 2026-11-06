# Model architecture

The Keras `Sequential` CNN is trained from scratch on **48 × 48 × 1** grayscale inputs.

![Two-panel architecture showing the input image, six convolutional tensors, three pooling stages and the dense classification head.](../assets/model-overview.svg)

**Figure 1.** (a) Convolutional feature extraction produces a **2 × 2 × 256** tensor. (b) Flattening and a **256-unit dense layer** produce seven Softmax outputs. Tensor volumes are schematic. [Image source](../assets/source/README.md).

## Feature extraction

The feature extractor has **three blocks**, each containing two convolutions, max pooling and Dropout.

All convolutions use a **3 × 3 kernel**, **ReLU**, **valid padding** and stride 1. With valid padding, each convolution reduces the image height and width by two pixels. Batch Normalization follows every convolution. A 2 × 2 max-pooling layer then halves the spatial dimensions, followed by Dropout at 0.25.

During training, Batch Normalization uses batch statistics to normalise each channel. Dropout randomly sets a proportion of its inputs to zero, so the following layers cannot rely on the same set of values at every update. These operations are included in all three blocks.

| Stage | Convolution outputs | Output after pooling |
| --- | --- | --- |
| Input | — | 48 × 48 × 1 |
| Block 1 | 46 × 46 × 32; 44 × 44 × 64 | 22 × 22 × 64 |
| Block 2 | 20 × 20 × 128; 18 × 18 × 128 | 9 × 9 × 128 |
| Block 3 | 7 × 7 × 256; 5 × 5 × 256 | 2 × 2 × 256 |

Spatial resolution decreases while the channel count grows from 32 to 256. The final tensor contains the learned features passed to the classifier.

The first block produces 64 feature maps at 22 × 22 pixels. The second reduces them to 9 × 9 with 128 channels, and the third produces 256 maps at 2 × 2. These sizes follow directly from the convolution and pooling settings shown in the table.

## Classifier

The 2 × 2 × 256 tensor is flattened to **1,024 values**. A dense layer with **256 ReLU units**, Batch Normalization and **Dropout 0.5** precedes the final **seven-unit Softmax** layer.

Flattening keeps all 1,024 feature values but arranges them as a vector. The dense layer combines those features for classification. Softmax converts the seven output values into scores that sum to one, and the largest score determines the predicted class.

The Softmax output follows the generator's class order: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## Layer shapes and parameters

The table follows the model definition and `model.summary()`, with the batch dimension omitted. The layer values are also available in [architecture.csv](../results/architecture.csv).

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| Conv2D, 32 filters | 46 × 46 × 32 | 320 |
| Batch Normalization | 46 × 46 × 32 | 128 |
| Conv2D, 64 filters | 44 × 44 × 64 | 18,496 |
| Batch Normalization | 44 × 44 × 64 | 256 |
| Max pooling 2 × 2 | 22 × 22 × 64 | 0 |
| Dropout 0.25 | 22 × 22 × 64 | 0 |
| Conv2D, 128 filters | 20 × 20 × 128 | 73,856 |
| Batch Normalization | 20 × 20 × 128 | 512 |
| Conv2D, 128 filters | 18 × 18 × 128 | 147,584 |
| Batch Normalization | 18 × 18 × 128 | 512 |
| Max pooling 2 × 2 | 9 × 9 × 128 | 0 |
| Dropout 0.25 | 9 × 9 × 128 | 0 |
| Conv2D, 256 filters | 7 × 7 × 256 | 295,168 |
| Batch Normalization | 7 × 7 × 256 | 1,024 |
| Conv2D, 256 filters | 5 × 5 × 256 | 590,080 |
| Batch Normalization | 5 × 5 × 256 | 1,024 |
| Max pooling 2 × 2 | 2 × 2 × 256 | 0 |
| Dropout 0.25 | 2 × 2 × 256 | 0 |
| Flatten | 1,024 | 0 |
| Dense, 256 ReLU units | 256 | 262,400 |
| Batch Normalization | 256 | 1,024 |
| Dropout 0.5 | 256 | 0 |
| Dense, 7 Softmax units | 7 | 1,799 |
| **Total** | | **1,394,183** |

**Trainable:** 1,391,943. **Non-trainable:** 2,240, corresponding to the Batch Normalization moving statistics.

For a convolution with a bias term, the count is `(kernel_height × kernel_width × input_channels + 1) × output_channels`. A dense layer has `(input_units + 1) × output_units`. Each Batch Normalization layer has four values per channel: two trainable parameters and two moving statistics. Pooling, Dropout and Flatten have no trainable parameters.

[Experiment and results](experiment.md) · [References](references.md) · [Overview](../README.md)
