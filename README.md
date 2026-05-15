# Deep Learning Image Classification: MNIST, CIFAR-10 & Fashion-MNIST

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

## Project Overview

This project explores image classification using neural networks built with TensorFlow/Keras across three progressively challenging benchmark datasets. Starting from a simple fully-connected network on handwritten digits, it advances to Convolutional Neural Networks with regularisation techniques applied to real-world colour photographs and fashion items.

The work demonstrates how dataset complexity directly drives architectural choices — from a 2-layer Dense network achieving ~98% accuracy on MNIST to a regularised 3-block CNN reaching ~75% on the considerably harder CIFAR-10 — and provides a hands-on study of overfitting, regularisation strategies, and model evaluation.

## Key Features

* **MNIST Handwritten Digit Classification (Task 1):** Training a fully-connected network (784 → 512 ReLU → 10 Softmax) on 70 000 grayscale images of handwritten digits using the `rmsprop` optimiser and `categorical_crossentropy` loss. The notebook covers data preprocessing (flatten, normalise, one-hot encode), training curve analysis, model serialisation to `.keras` format, and a real-world extension: testing the trained model on custom hand-drawn digit images with a dedicated preprocessing pipeline (grayscale conversion, resizing, colour inversion, normalisation).

* **CIFAR-10 Object Classification (Task 2):** Building and regularising a 3-block CNN to classify 32×32 colour photographs into 10 object categories. The notebook demonstrates the overfitting problem on a baseline model (~71% accuracy) and resolves it by applying four complementary regularisation techniques: Data Augmentation (`RandomFlip`, `RandomRotation`), Batch Normalisation, Dropout (0.25 per conv block, 0.5 before output), and Early Stopping — improving accuracy to ~75% with no overfitting.

* **Fashion-MNIST Clothing Classification (Task 3):** Classifying 28×28 grayscale clothing images into 10 categories using a 2-block CNN. This notebook contrasts with CIFAR-10 by deliberately avoiding data augmentation (clothing items have a natural fixed orientation) and instead relying on graduated Dropout (0.2 → 0.3 → 0.4), Batch Normalisation, and an `EarlyStopping` callback with `patience=5`. The regularised model achieves ~92% test accuracy, with validation accuracy consistently matching or exceeding training accuracy — a hallmark of effective regularisation.

## Technologies Used

* **Language:** Python 3.8+
* **Deep Learning:** TensorFlow 2.x / Keras (`Sequential`, `Dense`, `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`, `Flatten`)
* **Callbacks:** `EarlyStopping` with `restore_best_weights`
* **Data Augmentation:** Keras `RandomFlip`, `RandomRotation` preprocessing layers
* **Data Manipulation:** NumPy
* **Data Visualization:** Matplotlib
* **Environment:** Jupyter Notebook / Google Colab

## Datasets

* **MNIST:** 70 000 grayscale images (28×28) of handwritten digits 0–9 (60 000 train + 10 000 test). Loaded via `tensorflow.keras.datasets.mnist`. Also tested against 4 custom hand-drawn digit images stored in the `my_digits/` folder.
* **CIFAR-10:** 60 000 colour images (32×32, 3 channels) of 10 real-world object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (50 000 train + 10 000 test). Loaded via `tensorflow.keras.datasets.cifar10`.
* **Fashion-MNIST:** 70 000 grayscale images (28×28) of 10 clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot (60 000 train + 10 000 test). Loaded via `tensorflow.keras.datasets.fashion_mnist`.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arcctg/neural-image-classifiers.git
   cd neural-image-classifiers
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install numpy matplotlib tensorflow jupyter
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   *Open the corresponding `.ipynb` files in your browser and run cells sequentially.*

> **Note:** Notebooks are also runnable on [Google Colab](https://colab.research.google.com/) without any local setup — each notebook includes a Colab badge at the top.

## Results and Insights

* **Task 1 — MNIST (~98% accuracy):** The simple Dense network converges quickly in just 5 epochs with `rmsprop`, reaching ~97.7% test accuracy. Testing on custom hand-drawn digits revealed a key limitation of fully-connected architectures: they are sensitive to pixel distribution shifts. Freehand digits yielded only 1/4 correct predictions, while MNIST-style redrawn digits improved to 3/4 — highlighting the importance of colour inversion preprocessing and matched data distributions.

* **Task 2 — CIFAR-10 (~75% accuracy):** The baseline CNN showed clear overfitting by epoch 9. Applying all four regularisation techniques resolved this: validation accuracy (~75%) actually surpassed training accuracy (~70%) because Dropout and Data Augmentation are active only during training, intentionally making it harder. Early Stopping halted training at epoch 20, restoring the best weights from epoch 13. The model still struggles with visually similar pairs (cat/dog, automobile/truck), a known challenge for standard CNNs on this dataset.

* **Task 3 — Fashion-MNIST (~92% accuracy):** The regularised 2-block CNN with graduated Dropout and Batch Normalisation eliminated overfitting, as confirmed by validation accuracy matching or exceeding training accuracy. Early Stopping halted training at epoch 27, restoring best weights from epoch 22 (val_loss ≈ 0.2147). The most commonly confused classes are visually similar categories: Shirt vs. T-shirt/top, Pullover vs. Coat — pairs that overlap significantly in shape and texture.

### Cross-Task Comparison

| Dataset | Architecture | Test Accuracy | Key Challenge |
|---------|-------------|:------------:|--------------|
| MNIST (Task 1) | Dense 512 → 10 | ~98% | Low — clean digits, blank background |
| Fashion-MNIST (Task 3) | CNN (2 conv blocks) | ~92% | Medium — complex shapes, similar classes |
| CIFAR-10 (Task 2) | CNN (3 conv blocks) | ~75% | High — colour photos, varied backgrounds |

This progression illustrates how image complexity demands increasingly powerful architectures and regularisation strategies.

## License

This project is open-source and available under the [MIT License](LICENSE).
