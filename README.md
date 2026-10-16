# Fake Image Detection

A convolutional neural network (CNN) that classifies images as **real** or **AI-generated**, built with Python, OpenCV, and TensorFlow/Keras.

## Dataset
[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) from Kaggle: 60,000 real and 60,000 AI-generated 32x32 images for training, plus a separate test set. The dataset is not included in this repo. To reproduce, download it from Kaggle and place `archive.zip` in your working folder.

## Approach
1. **Preprocessing (OpenCV):** loaded images, resized to 32x32, normalized pixel values to 0-1.
2. **Baseline model:** a small CNN (2 convolution layers, max pooling, dropout, sigmoid output) trained on 10,000 images.
3. **Evaluation:** held out a validation split during training, then scored once on a separate test set of 2,000 images.
4. **Second experiment:** a deeper CNN with batch normalization and data augmentation, trained on 40,000 images.

## Results
| Model | Test accuracy |
|---|---|
| Baseline CNN | 85% |
| Deeper CNN + augmentation | 84% |

**Baseline classification report (test set):**

| Class | Precision | Recall |
|---|---|---|
| REAL | 0.93 | 0.76 |
| FAKE | 0.80 | 0.94 |

## Key findings
- The model is biased toward predicting "fake": it catches 94% of AI-generated images but wrongly flags about 24% of real ones.
- The deeper model with more data and augmentation did **not** improve accuracy. A likely reason is that augmentation such as image shifting can distort the fine pixel-level artifacts that fake-image detection relies on. [Add what you found if you ran the threshold or early-stopping experiments.]

## What I'd try next
- Tune the decision threshold using the validation set
- Train longer with early stopping
- Use the full 100,000-image training set
- Try transfer learning with a pretrained model

## Tech stack
Python, OpenCV, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib

## How to run
1. Download the dataset from the Kaggle link above.
2. Open `fake_image_detection.ipynb` in Google Colab.
3. Upload `archive.zip`, run the cells in order.
