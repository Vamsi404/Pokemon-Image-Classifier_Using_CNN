
# Pokémon Image Classifier 🐾

This project implements a simple neural network from scratch using NumPy to classify images of Pokémon (Pikachu, Bulbasaur, Meowth). The network is trained using gradient descent and backpropagation, with performance evaluated through confusion matrices and classification reports.

## Overview 📚

- **Project Goal:** Build and train a neural network to classify images of Pokémon characters.
- **Dataset:** Contains 40x40 pixel images of three Pokémon classes.
- **Model Architecture:** Three layers (input, two hidden layers, and output layer).
- **Optimization:** Trained using gradient descent with backpropagation.
- **Performance Metrics:** Evaluated with accuracy scores, confusion matrices, and classification reports.

## Table of Contents 📖

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation 🛠️

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/PokéClassifier-NN.git
    ```
2. Navigate to the project directory:
    ```bash
    cd PokéClassifier-NN
    ```
3. Install the required libraries:
    ```bash
    pip install numpy matplotlib keras visualize
    ```

## Dataset 📊

The dataset consists of 40x40 pixel images from three Pokémon classes:
- Pikachu
- Bulbasaur
- Meowth

Place your dataset in the following structure:
```
Dataset/
    ├── Pikachu/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── Bulbasaur/
    │   ├── img1.jpg
    │   ├── img2.jpg
    └── Meowth/
        ├── img1.jpg
        ├── img2.jpg
```

## Model Architecture 🧠

The neural network is designed with three layers:
- **Input Layer:** 4800 features (40x40x3)
- **Hidden Layers:** 100 neurons in the first hidden layer, 50 in the second hidden layer
- **Output Layer:** 3 neurons for classification (Pikachu, Bulbasaur, Meowth)

## Training 🎯

The model is trained using a learning rate of `0.0002` for `500` epochs. The loss function is cross-entropy, and softmax is used for multi-class classification.

```python
model = NeuralNetwork(input_size=4800, layers=[100, 50], output_size=3)
train(X, Y, model, epochs=500, learning_rate=0.0002)
```

## Evaluation 📝

Training and test accuracy are calculated, along with confusion matrices and classification reports for in-depth analysis.

**Example:**
```python
print("Train Accuracy: %.4f" % getAccuracy(X, Y, model))
print("Test Accuracy: %.4f" % getAccuracy(XTest, YTest, model))
```

## Visualizations 📈

The training loss over epochs is visualized using Matplotlib:

```python
plt.style.use("dark_background")
plt.title("Training Loss vs Epochs")
plt.plot(training_loss)
plt.show()
```

Additionally, sample misclassified images are displayed for error analysis.

## Results 🏆

- **Train Accuracy:** 98.31%
- **Test Accuracy:** 61.80%
- The model performs well on the training set but shows overfitting on the test set.

Confusion matrices and classification reports give further insights into per-class performance.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a pull request or open an issue.
