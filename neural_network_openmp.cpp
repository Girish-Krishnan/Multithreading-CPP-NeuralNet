#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <string>
#include <stdexcept>

using namespace std;

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// Function to parse command-line arguments
void parse_arguments(int argc, char* argv[], string& train_file, double& learning_rate, int& iterations, 
                     double& train_ratio, int& hidden_size, int& num_threads) {
    // Default values
    train_file = "data/train.csv";
    learning_rate = 0.1;
    iterations = 40;
    train_ratio = 0.8;
    hidden_size = 10;
    num_threads = 4;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--train_file" && i + 1 < argc) {
            train_file = argv[++i];
        } else if (arg == "--learning_rate" && i + 1 < argc) {
            learning_rate = stod(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = stoi(argv[++i]);
        } else if (arg == "--train_ratio" && i + 1 < argc) {
            train_ratio = stod(argv[++i]);
        } else if (arg == "--hidden_size" && i + 1 < argc) {
            hidden_size = stoi(argv[++i]);
        } else if (arg == "--num_threads" && i + 1 < argc) {
            num_threads = stoi(argv[++i]);
        } else {
            throw invalid_argument("Invalid argument: " + arg);
        }
    }
}

// Helper functions for matrix operations
vector<vector<double>> initialize_matrix(int rows, int cols, double min_val = -0.5, double max_val = 0.5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    
    vector<vector<double>> mat(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = dis(gen);
    return mat;
}

vector<double> initialize_vector(int size, double min_val = -0.5, double max_val = 0.5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);

    vector<double> vec(size);
    for (double &v : vec)
        v = dis(gen);
    return vec;
}

vector<double> relu(const vector<double>& x) {
    vector<double> res(x.size());
    #pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        res[i] = max(0.0, x[i]);
    }
    return res;
}

double exp_sum(const vector<double>& x) {
    return accumulate(x.begin(), x.end(), 0.0, [](double sum, double val) { return sum + exp(val); });
}

vector<double> softmax(const vector<double>& x) {
    vector<double> res(x.size());
    double total_exp = 0.0;

    #pragma omp parallel for reduction(+:total_exp)
    for (int i = 0; i < x.size(); ++i) {
        res[i] = exp(x[i]);
        total_exp += res[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        res[i] /= total_exp;
    }

    return res;
}

vector<double> one_hot(int label, int num_classes) {
    vector<double> one_hot_vector(num_classes, 0.0);
    one_hot_vector[label] = 1.0;
    return one_hot_vector;
}

vector<vector<double>> transpose(const vector<vector<double>>& mat) {
    int rows = mat.size(), cols = mat[0].size();
    vector<vector<double>> trans(cols, vector<double>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            trans[j][i] = mat[i][j];
    return trans;
}

vector<double> mat_vec_mult(const vector<vector<double>>& mat, const vector<double>& vec) {
    vector<double> result(mat.size(), 0.0);

    #pragma omp parallel for
    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }

    return result;
}

vector<double> vec_add(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());

    #pragma omp parallel for
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

vector<vector<double>> load_csv(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> data;
    string line;

    // Skip the header row
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;
        while (getline(ss, value, ',')) {
            try {
                row.push_back(stod(value));
            } catch (const invalid_argument& e) {
                cerr << "Invalid data in CSV: " << value << endl;
                exit(EXIT_FAILURE);
            }
        }
        data.push_back(row);
    }
    return data;
}

vector<int> get_predictions(const vector<vector<double>>& outputs) {
    vector<int> predictions(outputs.size());
    for (int i = 0; i < outputs.size(); ++i)
        predictions[i] = max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin();
    return predictions;
}

double get_accuracy(const vector<int>& predictions, const vector<int>& labels) {
    int correct = 0;
    for (int i = 0; i < predictions.size(); ++i)
        if (predictions[i] == labels[i])
            correct++;
    return static_cast<double>(correct) / labels.size();
}

void compute_forward(const vector<vector<double>>& X, const vector<vector<double>>& W1, const vector<double>& B1,
                     const vector<vector<double>>& W2, const vector<double>& B2,
                     vector<vector<double>>& Z1, vector<vector<double>>& A1, 
                     vector<vector<double>>& Z2, vector<vector<double>>& A2) {
    #pragma omp parallel for
    for (int i = 0; i < X.size(); ++i) {
        Z1[i] = vec_add(mat_vec_mult(W1, X[i]), B1);
        A1[i] = relu(Z1[i]);
        Z2[i] = vec_add(mat_vec_mult(W2, A1[i]), B2);
        A2[i] = softmax(Z2[i]);
    }
}

void compute_backprop(const vector<vector<double>>& X, const vector<int>& Y, const vector<vector<double>>& A1,
                      const vector<vector<double>>& A2, const vector<vector<double>>& Z1, const vector<vector<double>>& W2,
                      vector<vector<double>>& dW1, vector<double>& dB1,
                      vector<vector<double>>& dW2, vector<double>& dB2) {
    #pragma omp parallel for
    for (int i = 0; i < X.size(); ++i) {
        auto one_hot_Y = one_hot(Y[i], dW2.size());
        vector<double> dZ2(dW2.size());

        for (int j = 0; j < dW2.size(); ++j) {
            dZ2[j] = A2[i][j] - one_hot_Y[j];
            #pragma omp atomic
            dB2[j] += dZ2[j];
            for (int k = 0; k < W2[0].size(); ++k) {
                #pragma omp atomic
                dW2[j][k] += dZ2[j] * A1[i][k];
            }
        }

        vector<double> dZ1(W2[0].size(), 0.0);
        for (int j = 0; j < W2[0].size(); ++j) {
            for (int k = 0; k < dW2.size(); ++k) {
                dZ1[j] += dZ2[k] * W2[k][j];
            }
            dZ1[j] *= (Z1[i][j] > 0);
            #pragma omp atomic
            dB1[j] += dZ1[j];
            for (int k = 0; k < X[0].size(); ++k) {
                #pragma omp atomic
                dW1[j][k] += dZ1[j] * X[i][k];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Variables for input parameters
    string train_file;
    double learning_rate;
    int iterations;
    double train_ratio;
    int hidden_size;
    int num_threads;

    // Parse command-line arguments
    try {
        parse_arguments(argc, argv, train_file, learning_rate, iterations, train_ratio, hidden_size, num_threads);
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
        cerr << "Usage: " << argv[0] 
             << " [--train_file <file>] [--learning_rate <rate>] [--iterations <count>] [--train_ratio <ratio>] "
                "[--hidden_size <size>] [--num_threads <count>]" << endl;
        return EXIT_FAILURE;
    }

    // Load dataset
    auto data = load_csv(train_file);

    // Shuffle data
    random_shuffle(data.begin(), data.end());

    // Split data into training and validation sets
    int m = data.size();
    int n = data[0].size();
    int train_size = train_ratio * m;

    vector<vector<double>> train_data(data.begin(), data.begin() + train_size);
    vector<vector<double>> val_data(data.begin() + train_size, data.end());

    // Extract features and labels
    vector<vector<double>> X_train(train_size, vector<double>(n - 1));
    vector<int> Y_train(train_size);
    for (int i = 0; i < train_size; ++i) {
        Y_train[i] = data[i][0];
        for (int j = 1; j < n; ++j)
            X_train[i][j - 1] = data[i][j] / 255.0;
    }

    vector<vector<double>> X_val(val_data.size(), vector<double>(n - 1));
    vector<int> Y_val(val_data.size());
    for (int i = 0; i < val_data.size(); ++i) {
        Y_val[i] = val_data[i][0];
        for (int j = 1; j < n; ++j)
            X_val[i][j - 1] = val_data[i][j] / 255.0;
    }

    // Initialize weights and biases
    auto W1 = initialize_matrix(hidden_size, INPUT_SIZE);
    auto B1 = initialize_vector(hidden_size);
    auto W2 = initialize_matrix(OUTPUT_SIZE, hidden_size);
    auto B2 = initialize_vector(OUTPUT_SIZE);

    // Training loop
    for (int iter = 0; iter < iterations; ++iter) {
        vector<vector<double>> A1(train_size, vector<double>(hidden_size));
        vector<vector<double>> Z1(train_size, vector<double>(hidden_size));
        vector<vector<double>> A2(train_size, vector<double>(OUTPUT_SIZE));
        vector<vector<double>> Z2(train_size, vector<double>(OUTPUT_SIZE));

        // Parallelize forward propagation
        compute_forward(X_train, W1, B1, W2, B2, Z1, A1, Z2, A2);

        // Parallelize backpropagation
        vector<vector<double>> dW2(OUTPUT_SIZE, vector<double>(hidden_size, 0.0));
        vector<double> dB2(OUTPUT_SIZE, 0.0);
        vector<vector<double>> dW1(hidden_size, vector<double>(INPUT_SIZE, 0.0));
        vector<double> dB1(hidden_size, 0.0);

        compute_backprop(X_train, Y_train, A1, A2, Z1, W2, dW1, dB1, dW2, dB2);

        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            for (int k = 0; k < hidden_size; ++k) {
                W2[j][k] -= learning_rate * dW2[j][k] / train_size;
            }
            B2[j] -= learning_rate * dB2[j] / train_size;
        }

        for (int j = 0; j < hidden_size; ++j) {
            for (int k = 0; k < INPUT_SIZE; ++k) {
                W1[j][k] -= learning_rate * dW1[j][k] / train_size;
            }
            B1[j] -= learning_rate * dB1[j] / train_size;
        }

        if (iter % 20 == 0) {
            // Compute training accuracy
            vector<int> train_predictions = get_predictions(A2);
            double train_accuracy = get_accuracy(train_predictions, Y_train);

            // Compute validation accuracy
            vector<vector<double>> val_A1(X_val.size(), vector<double>(hidden_size));
            vector<vector<double>> val_Z1(X_val.size(), vector<double>(hidden_size));
            vector<vector<double>> val_A2(X_val.size(), vector<double>(OUTPUT_SIZE));
            vector<vector<double>> val_Z2(X_val.size(), vector<double>(OUTPUT_SIZE));

            compute_forward(X_val, W1, B1, W2, B2, val_Z1, val_A1, val_Z2, val_A2);

            vector<int> val_predictions = get_predictions(val_A2);
            double val_accuracy = get_accuracy(val_predictions, Y_val);

            cout << "Iteration: " << iter
                << ", Training Accuracy: " << train_accuracy
                << ", Validation Accuracy: " << val_accuracy << endl;
        }
    }

    cout << "Training complete!" << endl;

    return 0;
}
