# Training a Neural Network from Scratch using C++ and Multithreading

Girish Krishnan | [LinkedIn](https://www.linkedin.com/in/girk/) | [GitHub](https://github.com/Girish-Krishnan)

___

![Neural Net](./media/architecture.png)

![Neurons](./media/neurons.png)

This project is a C++ implementation of a Neural Network from scratch. The Neural Network is trained using the MNIST dataset to classify handwritten digits. The key reference for this project is Dr. Sreedath Panat's tutorial on implementing a Neural Network from scratch in Python. The tutorial can be found on YouTube [here](https://www.youtube.com/watch?v=A83BbHFoKb8).  

However, this project is implemented in **C++** and uses **multithreading** to speed up the training process. Another key challenge using C++ is implementing many of the matrix operations that are readily available in Python libraries such as NumPy, and parallelizing these operations using multithreading.

---

## Mathematical Background

### Dataset: MNIST Handwritten Digits
The MNIST dataset consists of grayscale images of handwritten digits (0-9). Each image has dimensions of $28 \times 28$ pixels, and the dataset contains 10 output classes (one for each digit).

- Each image is represented as a vector of 784 dimensions by flattening the $28 \times 28$ grid into a single column vector:
  $$\mathbf{x} \in \mathbb{R}^{784}$$.

- The label for each image is represented as a one-hot encoded vector:
  $$\mathbf{y} \in \mathbb{R}^{10}, \quad \text{where } y_i = 1 \text{ if the image corresponds to digit } i, \text{ and } y_j = 0 \text{ for } j \neq i.$$

The dataset is split into training and validation sets:
- Training set: $\mathbf{X}_{\text{train}} \in \mathbb{R}^{784 \times N}$, where $N$ is the number of training samples.
- Validation set: $\mathbf{X}_{\text{val}} \in \mathbb{R}^{784 \times M}$, where $M$ is the number of validation samples.

### Neural Network Architecture
The neural network consists of:
1. An input layer with 784 neurons (one for each pixel).
2. A hidden layer with $H$ neurons, using the ReLU activation function.
3. An output layer with 10 neurons, using the softmax activation function.

#### Parameters
- $\mathbf{W}_1 \in \mathbb{R}^{H \times 784}$: Weights for the first layer.
- $\mathbf{b}_1 \in \mathbb{R}^{H \times 1}$: Bias for the first layer.
- $\mathbf{W}_2 \in \mathbb{R}^{10 \times H}$: Weights for the second layer.
- $\mathbf{b}_2 \in \mathbb{R}^{10 \times 1}$: Bias for the second layer.

### Forward Propagation
1. Compute the pre-activation of the hidden layer:
   $$\mathbf{Z}_1 = \mathbf{W}_1 \mathbf{X} + \mathbf{b}_1, \quad \mathbf{Z}_1 \in \mathbb{R}^{H \times N}$$.

2. Apply the ReLU activation function:
   $$\mathbf{A}_1 = \text{ReLU}(\mathbf{Z}_1), \quad \text{where } \text{ReLU}(z) = \max(0, z)$$.

3. Compute the pre-activation of the output layer:
   $$\mathbf{Z}_2 = \mathbf{W}_2 \mathbf{A}_1 + \mathbf{b}_2, \quad \mathbf{Z}_2 \in \mathbb{R}^{10 \times N}$$.

4. Apply the softmax activation function to obtain probabilities:
   $$\mathbf{A}_2 = \text{softmax}(\mathbf{Z}_2), \quad \text{where } [\text{softmax}(\mathbf{Z}_2)]_{ij} = \frac{\exp(\mathbf{Z}_2[i,j])}{\sum_{k=1}^{10} \exp(\mathbf{Z}_2[k,j])}$$.

### Loss Function
The cross-entropy loss is used to measure the error between predicted probabilities $\mathbf{A}_2$ and true labels $\mathbf{Y}$:
$$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{10} y_{ij} \log(a_{2,ij})$$,
where $y_{ij}$ is the true label and $a_{2,ij}$ is the predicted probability for sample $i$ and class $j$.

### Backward Propagation
1. Compute the gradient of the loss with respect to $\mathbf{Z}_2$:
   $$\mathbf{dZ}_2 = \mathbf{A}_2 - \mathbf{Y}$$.

2. Compute gradients for $\mathbf{W}_2$ and $\mathbf{b}_2$:
   $$\mathbf{dW}_2 = \frac{1}{N} \mathbf{dZ}_2 \mathbf{A}_1^T, \quad \mathbf{db}_2 = \frac{1}{N} \sum_{i=1}^{N} \mathbf{dZ}_2[:,i]$$.

3. Compute the gradient of the loss with respect to $\mathbf{Z}_1$:
   $$\mathbf{dZ}_1 = \mathbf{W}_2^T \mathbf{dZ}_2 \odot \text{ReLU}'(\mathbf{Z}_1)$$,
   where $\text{ReLU}'(z) = 1$ if $z > 0$, else $0$.

4. Compute gradients for $\mathbf{W}_1$ and $\mathbf{b}_1$:
   $$\mathbf{dW}_1 = \frac{1}{N} \mathbf{dZ}_1 \mathbf{X}^T, \quad \mathbf{db}_1 = \frac{1}{N} \sum_{i=1}^{N} \mathbf{dZ}_1[:,i]$$.

### Parameter Update
Update the parameters using gradient descent:
$$\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \alpha \mathbf{dW}_1, \quad \mathbf{b}_1 \leftarrow \mathbf{b}_1 - \alpha \mathbf{db}_1$$,

$$\mathbf{W}_2 \leftarrow \mathbf{W}_2 - \alpha \mathbf{dW}_2, \quad \mathbf{b}_2 \leftarrow \mathbf{b}_2 - \alpha \mathbf{db}_2$$,

where $\alpha$ is the learning rate.

---

## Implementation Details

This repository includes three implementations of a simple neural network for the MNIST dataset, each demonstrating different approaches to parallelization:

1. **`neural_network.cpp`**: A sequential implementation of the neural network.
2. **`neural_network_thread.cpp`**: A multi-threaded implementation using the C++ threading library.
3. **`neural_network_openmp.cpp`**: A parallelized implementation using OpenMP.

Each implementation trains a simple neural network with one hidden layer using gradient descent and computes the accuracy on a validation set.

---

## Command-Line Arguments

The implementations allow customization of parameters through command-line arguments. Below is the list of supported arguments:

| Argument           | Default Value       | Description                                                                 |
|--------------------|---------------------|-----------------------------------------------------------------------------|
| `--train_file`     | `data/train.csv`    | Path to the training dataset in CSV format.                                 |
| `--learning_rate`  | `0.1`               | Learning rate for gradient descent.                                         |
| `--iterations`     | `40`                | Number of iterations to train the model.                                    |
| `--train_ratio`    | `0.8`               | Ratio of the dataset to use for training. The rest is used for validation.  |
| `--hidden_size`    | `10`                | Number of neurons in the hidden layer.                                      |
| `--num_threads`    | `4` (threaded/OpenMP) | Number of threads to use (only applicable to threaded or OpenMP versions). |

---

## Compilation Instructions

To compile each implementation, use the following commands:

### Sequential Implementation
```bash
$ g++ -o neural_network neural_network.cpp -std=c++11
```

### Threaded Implementation
```bash
$ g++ -o neural_network_thread neural_network_thread.cpp -std=c++11 -pthread
```

### OpenMP Implementation
```bash
$ g++ -o neural_network_openmp neural_network_openmp.cpp -std=c++11 -fopenmp
```

Note: if you're using MacOS, you may need to install `llvm` to enable OpenMP support.

---

## Running the Programs

Each program accepts the same arguments. Below is an example of how to run the implementations:

### Example Command
```bash
$ ./neural_network --train_file ./data/train.csv --learning_rate 0.1 --iterations 100 --train_ratio 0.8 --hidden_size 10
```

For the threaded or OpenMP versions, you can also specify the number of threads:

```bash
$ ./neural_network_thread --train_file ./data/train.csv --learning_rate 0.1 --iterations 100 --train_ratio 0.8 --hidden_size 10 --num_threads 4
```

```bash
$ ./neural_network_openmp --train_file ./data/train.csv --learning_rate 0.1 --iterations 100 --train_ratio 0.8 --hidden_size 10 --num_threads 4
```

---

## Example Output

Below is an example of the output generated by running the program.

```plaintext
Iteration: 0, Training Accuracy: 0.094881, Validation Accuracy: 0.100238
Iteration: 20, Training Accuracy: 0.298512, Validation Accuracy: 0.305238
Iteration: 40, Training Accuracy: 0.42997, Validation Accuracy: 0.432976
Iteration: 60, Training Accuracy: 0.518423, Validation Accuracy: 0.520357
Iteration: 80, Training Accuracy: 0.599345, Validation Accuracy: 0.592381
Training complete!
```

The output displays the training and validation accuracy at regular intervals during the training process.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

