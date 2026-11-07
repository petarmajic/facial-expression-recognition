# Data and preprocessing

I loaded the [FER2013 directory dataset](https://www.kaggle.com/datasets/msambare/fer2013) through `kagglehub.dataset_download("msambare/fer2013")`. Of the **35,887 images**, **28,709 are in `train/`** and **7,178 are in `test/`**.

## Dataset examples

![FER2013 examples of angry, disgust, fear, happy, neutral, sad and surprise.](../assets/dataset-examples.svg)

**Figure 1.** I used these seven examples in Figure 3.2 of my thesis. [Image source](../assets/source/README.md).

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

The *happy* class accounts for about **25.1%** of the full training directory and *disgust* for **1.5%**. I counted the files before selecting subsets. A mistake on a *disgust* image counts just as much as one on a *happy* image, but there are far fewer *disgust* images to influence overall accuracy. [Counts for both directories](../results/class_counts.csv).

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

I divided pixel values by 255 so both subsets used the same 0–1 scale. Only the training generator shifted and flipped images. This let the model see a face in different positions while the validation images kept their original orientation.

## The split actually used

I set `validation_split=0.2` on both generators. Each generator applied that fraction within the class folders of the directory it loaded.

| Generator | Directory | Subset | Images | Role |
| --- | --- | --- | ---: | --- |
| Training | `train/` | `training` | **22,968** | Gradient updates |
| Validation | `test/` | `validation` | **1,432** | Per-epoch evaluation and checkpoint selection |

The training generator took 80% of each class in `train/`. The validation generator took 20% of each class in `test/`. I used the resulting 1,432 validation images to select the checkpoint that reached **65.22%**. The notebook records no separate final test evaluation.

[Experiment](experiment.md) · [Execution](reproduction.md) · [Overview](../README.md)
