# Data and preprocessing

The notebook loads the [FER2013 directory dataset](https://www.kaggle.com/datasets/msambare/fer2013) through `kagglehub.dataset_download("msambare/fer2013")`. It contains **28,709 images in `train/`** and **7,178 in `test/`**, for **35,887 images** in total.

## Dataset examples

![FER2013 examples of angry, disgust, fear, happy, neutral, sad and surprise.](../assets/dataset-examples.svg)

**Figure 1.** Examples of the seven expression classes, from thesis Figure 3.2. [Image source](../assets/source/README.md).

## Images per class

![Class counts in the complete FER2013 training directory.](../assets/class-distribution.svg)

| Label | Source `train/` | Source `test/` |
| --- | ---: | ---: |
| angry | 3,995 | 958 |
| disgust | 436 | 111 |
| fear | 4,097 | 1,024 |
| happy | 7,215 | 1,774 |
| neutral | 4,965 | 1,233 |
| sad | 4,830 | 1,247 |
| surprise | 3,171 | 831 |
| **Total** | **28,709** | **7,178** |

Happy accounts for approximately **25.1%** of the training directory, compared with **1.5%** for disgust. The chart and [class_counts.csv](../results/class_counts.csv) describe these full directories, before subset selection.

The classes therefore contribute very different numbers of training examples. These counts give context to the overall accuracy, which combines predictions from all seven classes into one number.

## Image preprocessing

| Setting | Training | Validation |
| --- | --- | --- |
| Loader | Keras `ImageDataGenerator.flow_from_directory` | Same |
| Target size | 48 × 48 | 48 × 48 |
| Colour mode | Grayscale | Grayscale |
| Pixel rescaling | `1./255` | `1./255` |
| Labels | Categorical / one-hot | Categorical / one-hot |
| Batch size | 64 | 64 |
| Width / height shifts | Up to 10% of the image dimension | None |
| Horizontal flips | Enabled | Disabled |
| Shuffle | Default `True` | Default `True` |
| Random seed | Not set | Not set |

The loader converts each image to grayscale, resizes it to 48 × 48 and divides pixel values by 255. During training, random shifts and horizontal flips provide variations of the input images. Validation uses the same size and pixel scale without those augmentations.

## The split actually used

Both generators use `validation_split=0.2`, applied within their respective class directories:

| Generator | Directory | Subset | Images | Role |
| --- | --- | --- | ---: | --- |
| Training | `train/` | `training` | **22,968** | Gradient updates |
| Validation | `test/` | `validation` | **1,432** | Per-epoch evaluation and checkpoint selection |

Training uses 80% of the source training directory and validation uses 20% of the source test directory. Keras selects these fractions within each class directory. The folder names alone therefore do not describe the experiment's actual training and validation sizes.

The reported **65.22%** is measured on the 1,432-image validation subset. That subset was also used to choose the saved weights. A separate final test evaluation is not recorded.

[Experiment](experiment.md) · [Execution](reproduction.md) · [Overview](../README.md)
