# MNIST Neural Network Trainer

A real-time, interactive neural network trainer and visualizer built with Blazor WebAssembly and ASP.NET Core. Train a handwritten digit classifier on MNIST-style images and watch the network learn in your browser.

Most examples do recognition using an existing model. This app is focused on the learning process and uses it's output for learning.

![Architecture: 784 → 128 → 10 with Sigmoid activation]

## Features

- **Live network visualization** — SVG-rendered nodes for input, hidden, and output layers with color-coded activations and weighted connections
- **Click-to-inspect nodes** — view bias, pre-activation, sigmoid activation, delta, and weight statistics for any neuron
- **Real-time training** — backpropagation with stochastic gradient descent, running entirely in the browser via WebAssembly
- **Training controls** — play/pause, single-step, forward-only pass, and full network reset
- **Adjustable parameters** — sliders for learning rate, training speed (steps per tick), and tick interval
- **Training progress chart** — loss and accuracy plotted per epoch
- **Prediction chart** — stacked correct vs. incorrect predictions per 100-sample window
- **Current sample preview** — displays the 28x28 input image with its label and the network's prediction
- **Model info panel** — shows total parameters, weights, biases, memory size, and architecture
- **Model export/import** — save trained weights as JSON, reload to resume training later
- **JPG/PNG folder loading** — drop images into digit-labeled subfolders and load directly from disk
- **IDX file upload** — also supports the classic MNIST binary format as a fallback

## Architecture

The network uses a fully-connected feed-forward architecture:

| Layer | Neurons | Activation | Description |
|-------|---------|------------|-------------|
| Input | 784 | — | One neuron per pixel (28x28 image) |
| Hidden | 128 | Sigmoid | Densely connected to all inputs |
| Output | 10 | Sigmoid | One neuron per digit class (0–9) |

**Total parameters:** 101,770 (100,480 weights + 1,290 biases) — approximately 796 KB in memory.

Weights are initialized using Xavier initialization. Training uses per-sample stochastic gradient descent with binary cross-entropy loss.

## Getting Started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)

### Setup

```bash
git clone <repo-url>
cd "Mnist NN Training"
```

### Prepare Training Data

Place your 28x28 pixel JPG or PNG images into digit-labeled subfolders:

```
TrainingData/
├── 0/    ← images of digit 0
├── 1/    ← images of digit 1
├── 2/
├── ...
└── 9/    ← images of digit 9
```

The server reads these on demand and converts them to grayscale pixel arrays.

### Run

```bash
cd MnistNNTraining.Server
dotnet run
```

Open the URL shown in the terminal (typically `https://localhost:5001` or `http://localhost:5000`). The app will auto-detect images in your `TrainingData/` folder.

## Controls Reference

### Playback

| Control | Description |
|---------|-------------|
| **Play / Pause** | Start or pause continuous training. While playing, the network processes samples and updates the visualization each tick. |
| **Step** | Process a single training sample (forward pass + backpropagation + weight update). Only available when paused. |
| **Forward Only** | Run a forward pass on the current sample without training. Useful for inspecting what the network predicts without changing weights. |
| **Reset** | Reinitialize all weights and biases to random values. Clears all training progress, charts, and statistics. |

### Sliders

| Slider | Range | Default | Description |
|--------|-------|---------|-------------|
| **Speed (steps/tick)** | 1–100 | 1 | How many training samples to process per timer tick. Higher values train faster but update the visualization less frequently per sample. |
| **Learning Rate** | 0.001–2.0 | 0.5 | Controls how much weights change on each update. Higher values learn faster but risk overshooting and instability. Lower values are more stable but slower. A good starting range is 0.1–1.0 for this network. |
| **Tick Interval (ms)** | 16–1000 | 100 | Milliseconds between each training tick. At 16ms the UI updates ~60 times per second (fastest). At 1000ms it updates once per second (slowest). Combined with steps/tick, these two sliders give full control over training speed. |
| **Max images per digit** | 10–1000 | 100 | Shown on the load screen. Caps how many images are loaded from each digit folder. 100 per digit = 1,000 total samples. Increase for better training, decrease for faster loading. |

### Statistics Display

| Stat | Meaning |
|------|---------|
| **Epoch** | How many full passes through the training data have completed. |
| **Sample** | Current position within the current epoch (e.g., 450 / 1000). |
| **Total Processed** | Cumulative number of samples the network has trained on. |
| **Avg Loss** | Average cross-entropy loss for the current epoch. Should decrease over time. |
| **Last Accuracy** | Percentage of correct predictions in the most recently completed epoch. |

### Visualization

- **Node colors** represent activation values: blue (low, near 0) through white (0.5) to orange (high, near 1).
- **Connection colors** represent weight sign: green for positive weights, red for negative. Opacity indicates magnitude.
- **Output nodes** are labeled 0–9. The predicted digit (highest activation) is highlighted with a green border.
- **Click any node** to open a detail panel showing its bias, pre-activation value, sigmoid activation, backpropagation delta, and weight statistics.

### Charts

- **Training Progress** — plots average loss (red) and accuracy percentage (green) at the end of each epoch. As the network learns, loss decreases and accuracy increases.
- **Predictions** — stacked bar chart showing correct (green) vs. incorrect (red) predictions per 100-sample window. Provides a more granular view of learning progress than the per-epoch chart.

### Model Export / Import

- **Export Model (JSON)** — downloads a JSON file containing all learned weights, biases, the learning rate, epoch count, and layer sizes. This is a complete snapshot of the trained model's parameters.
- **Import Model** — upload a previously exported JSON file to restore weights and resume training. The architecture must match (784 → 128 → 10).

The exported JSON can also be parsed by other applications to use the trained weights for inference.

## Project Structure

```
Mnist NN Training/
├── MnistNNTraining.sln
├── README.md
├── TrainingData/                    ← your training images (0-9 subfolders)
│
├── MnistNNTraining/                 ← Blazor WebAssembly client
│   ├── Program.cs                   ← app entry point, service registration
│   ├── Pages/
│   │   └── Index.razor              ← main page (data loading, layout, timer)
│   ├── Components/
│   │   ├── NetworkVisualization.razor  ← SVG network diagram
│   │   ├── TrainingControls.razor     ← play/pause, sliders, stats
│   │   ├── ImagePreview.razor         ← 28x28 sample display
│   │   ├── LossChart.razor            ← loss/accuracy line chart
│   │   └── PredictionChart.razor      ← correct/incorrect bar chart
│   ├── Models/
│   │   ├── NeuralNetwork.cs         ← forward pass, backprop, export/import
│   │   ├── Layer.cs                  ← weights, biases, activations per layer
│   │   └── MnistLoader.cs           ← IDX and binary format parsers
│   └── Services/
│       └── TrainingService.cs       ← training loop, state, statistics
│
└── MnistNNTraining.Server/          ← ASP.NET Core host
    ├── Program.cs                   ← server setup, static files, API routing
    └── Controllers/
        └── DataController.cs        ← reads JPG/PNG images, serves to client
```

## How It Works

1. **Data loading** — the server reads JPG/PNG files from `TrainingData/0-9/`, converts each to a 28x28 grayscale byte array using ImageSharp, and sends them to the browser as a compact binary payload.

2. **Forward pass** — pixel values (normalized to 0–1) feed into the input layer. Each hidden and output neuron computes a weighted sum of its inputs plus a bias, then applies the sigmoid function: `σ(x) = 1 / (1 + e^(-x))`.

3. **Loss calculation** — binary cross-entropy loss is computed against a one-hot target vector (e.g., digit 3 → `[0,0,0,1,0,0,0,0,0,0]`).

4. **Backpropagation** — error deltas propagate backward through the network. Each weight and bias is adjusted proportionally to its contribution to the error, scaled by the learning rate.

5. **Visualization** — the UI re-renders after each tick, showing updated activations, connection weights, and prediction results in real time.

## License

MIT

Guido Adam - 2026
