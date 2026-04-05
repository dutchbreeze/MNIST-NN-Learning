using System.Text.Json;

namespace MnistNNTraining.Models;

/// <summary>
/// A fully-connected feed-forward neural network with sigmoid activation
/// and backpropagation training.
/// Architecture: 784 → 128 → 10
/// </summary>
public class NeuralNetwork
{
    public List<Layer> Layers { get; private set; } = new();
    public double LearningRate { get; set; } = 0.5;
    public int Epoch { get; private set; }
    public int TotalSamplesProcessed { get; private set; }
    public double LastLoss { get; private set; }
    public double Accuracy { get; private set; }

    private readonly Random _rng;
    private int _correctInEpoch;
    private int _samplesInEpoch;

    public NeuralNetwork(int seed = 42)
    {
        _rng = new Random(seed);
        InitializeNetwork();
    }

    private void InitializeNetwork()
    {
        Layers.Clear();

        // Input layer (784 neurons — one per pixel)
        Layers.Add(new Layer(784, 0, LayerType.Input, _rng));

        // Hidden layer (128 neurons, 784 inputs each)
        Layers.Add(new Layer(128, 784, LayerType.Hidden, _rng));

        // Output layer (10 neurons — one per digit, 128 inputs each)
        Layers.Add(new Layer(10, 128, LayerType.Output, _rng));
    }

    public void Reset()
    {
        InitializeNetwork();
        Epoch = 0;
        TotalSamplesProcessed = 0;
        LastLoss = 0;
        Accuracy = 0;
        _correctInEpoch = 0;
        _samplesInEpoch = 0;
    }

    /// <summary>
    /// Forward pass through the network. Returns output activations.
    /// </summary>
    public double[] Forward(double[] input)
    {
        if (input.Length != 784)
            throw new ArgumentException("Input must have 784 values (28x28 pixels).");

        // Set input layer activations
        for (int i = 0; i < input.Length; i++)
            Layers[0].Activations[i] = input[i];

        // Propagate through hidden and output layers
        for (int l = 1; l < Layers.Count; l++)
        {
            var prev = Layers[l - 1];
            var curr = Layers[l];

            for (int j = 0; j < curr.Size; j++)
            {
                double sum = curr.Biases[j];
                for (int i = 0; i < prev.Size; i++)
                {
                    sum += prev.Activations[i] * curr.Weights[j][i];
                }
                curr.PreActivations[j] = sum;
                curr.Activations[j] = Sigmoid(sum);
            }
        }

        return Layers[^1].Activations;
    }

    /// <summary>
    /// Train on a single sample using backpropagation.
    /// Returns the loss for this sample.
    /// </summary>
    public double TrainSingle(double[] input, int label)
    {
        // Forward pass
        double[] output = Forward(input);

        // Build target (one-hot)
        double[] target = new double[10];
        target[label] = 1.0;

        // Calculate cross-entropy loss
        double loss = 0;
        for (int i = 0; i < 10; i++)
        {
            double o = Math.Clamp(output[i], 1e-15, 1 - 1e-15);
            loss -= target[i] * Math.Log(o) + (1 - target[i]) * Math.Log(1 - o);
        }
        LastLoss = loss;

        // Track accuracy
        int predicted = Array.IndexOf(output, output.Max());
        if (predicted == label) _correctInEpoch++;
        _samplesInEpoch++;
        TotalSamplesProcessed++;

        // ---- Backpropagation ----

        // Output layer deltas
        var outputLayer = Layers[^1];
        for (int j = 0; j < outputLayer.Size; j++)
        {
            double o = outputLayer.Activations[j];
            outputLayer.Deltas[j] = (o - target[j]) * SigmoidDerivative(outputLayer.PreActivations[j]);
        }

        // Hidden layer deltas
        for (int l = Layers.Count - 2; l >= 1; l--)
        {
            var curr = Layers[l];
            var next = Layers[l + 1];

            for (int j = 0; j < curr.Size; j++)
            {
                double sum = 0;
                for (int k = 0; k < next.Size; k++)
                {
                    sum += next.Deltas[k] * next.Weights[k][j];
                }
                curr.Deltas[j] = sum * SigmoidDerivative(curr.PreActivations[j]);
            }
        }

        // Update weights and biases
        for (int l = 1; l < Layers.Count; l++)
        {
            var curr = Layers[l];
            var prev = Layers[l - 1];

            for (int j = 0; j < curr.Size; j++)
            {
                curr.Biases[j] -= LearningRate * curr.Deltas[j];
                for (int i = 0; i < prev.Size; i++)
                {
                    curr.Weights[j][i] -= LearningRate * curr.Deltas[j] * prev.Activations[i];
                }
            }
        }

        return loss;
    }

    /// <summary>
    /// Call at the end of each epoch to update accuracy stats.
    /// </summary>
    public void EndEpoch()
    {
        Epoch++;
        Accuracy = _samplesInEpoch > 0 ? (double)_correctInEpoch / _samplesInEpoch * 100.0 : 0;
        _correctInEpoch = 0;
        _samplesInEpoch = 0;
    }

    public int Predict(double[] input)
    {
        double[] output = Forward(input);
        return Array.IndexOf(output, output.Max());
    }

    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -500, 500)));

    public static double SigmoidDerivative(double x)
    {
        double s = Sigmoid(x);
        return s * (1.0 - s);
    }

    /// <summary>
    /// Export model to JSON.
    /// </summary>
    public string ExportToJson()
    {
        var data = new NetworkData
        {
            LearningRate = LearningRate,
            Epoch = Epoch,
            LayerSizes = Layers.Select(l => l.Size).ToArray(),
            LayerWeights = Layers.Skip(1).Select(l => l.Weights.Select(w => w.ToArray()).ToArray()).ToArray(),
            LayerBiases = Layers.Skip(1).Select(l => l.Biases.ToArray()).ToArray()
        };
        return JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>
    /// Import model from JSON.
    /// </summary>
    public void ImportFromJson(string json)
    {
        var data = JsonSerializer.Deserialize<NetworkData>(json);
        if (data == null) return;

        LearningRate = data.LearningRate;
        Epoch = data.Epoch;

        for (int l = 1; l < Layers.Count; l++)
        {
            var layer = Layers[l];
            var wIdx = l - 1;
            for (int j = 0; j < layer.Size; j++)
            {
                layer.Biases[j] = data.LayerBiases[wIdx][j];
                for (int i = 0; i < layer.Weights[j].Length; i++)
                    layer.Weights[j][i] = data.LayerWeights[wIdx][j][i];
            }
        }
    }

    private class NetworkData
    {
        public double LearningRate { get; set; }
        public int Epoch { get; set; }
        public int[] LayerSizes { get; set; } = Array.Empty<int>();
        public double[][][] LayerWeights { get; set; } = Array.Empty<double[][]>();
        public double[][] LayerBiases { get; set; } = Array.Empty<double[]>();
    }
}
