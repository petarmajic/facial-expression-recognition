# Model architecture

The implementation uses a Keras `Sequential` CNN with an input shape of **48 × 48 × 1**. It learns features directly from the images; the notebook does not load pretrained weights.

![Two-panel architecture showing the input image, six convolutional tensors, three pooling stages and the dense classification head.](../assets/model-overview.svg)

**Figure 1.** (a) The convolutional feature extractor maps a grayscale image to a **2 × 2 × 256** tensor. (b) Flattening and a **256-unit dense layer** produce seven Softmax outputs. Tensor volumes are schematic and are not activation visualisations. The input image is an illustrative FER2013 example from the thesis montage; [source and crop details](../assets/source/README.md) are recorded separately.

## Feature extraction

For this explanation, the six convolutional layers are grouped into **three feature blocks**, each containing two convolutions, a pooling operation and Dropout. This grouping follows the implementation, rather than the five-block wording in the thesis text.

All convolutions use a **3 × 3 kernel**, **ReLU**, default **valid padding**, and stride 1. Batch Normalization follows every convolution. A 2 × 2 max-pooling layer then reduces spatial size, followed by Dropout at 0.25.

| Stage | Convolution outputs | Output after pooling |
| --- | --- | --- |
| Input | — | 48 × 48 × 1 |
| Block 1 | 46 × 46 × 32; 44 × 44 × 64 | 22 × 22 × 64 |
| Block 2 | 20 × 20 × 128; 18 × 18 × 128 | 9 × 9 × 128 |
| Block 3 | 7 × 7 × 256; 5 × 5 × 256 | 2 × 2 × 256 |

The number of channels increases as spatial resolution decreases. This gives the classifier a learned representation of the image without manually specified geometric or texture descriptors. The notebook does not include feature-map visualisations, so individual channels are not assigned a particular facial feature.

## Classifier

The 2 × 2 × 256 tensor is flattened to **1,024 values**. A dense layer with **256 ReLU units**, Batch Normalization and **Dropout 0.5** precedes the final **seven-unit Softmax** layer.

The output order follows the generator's alphabetical class order: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. Softmax gives seven scores summing to one; calibration of these scores was not evaluated.

## Layer shapes and parameters

The batch dimension is omitted below. The counts are derived from the code in notebook cell 3 and agree with its saved `model.summary()` output. The machine-readable table is [architecture.csv](../results/architecture.csv).

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

## What the run establishes

The saved outputs document one training run of this architecture. There are no ablations isolating the effect of Batch Normalization, Dropout, filter counts or augmentation. Their presence can be described precisely; an accuracy gain from any one component cannot be attributed from this experiment alone.

[Experiment and results](experiment.md) · [Evidence ledger](evidence.md) · [Back to overview](../README.md)
