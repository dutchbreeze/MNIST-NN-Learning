namespace MnistNNTraining.Models;

public enum LayerType
{
    Input,
    Hidden,
    Output
}

/// <summary>
/// A single layer in the neural network.
/// </summary>
public class Layer
{
    public int Size { get; }
    public int InputSize { get; }
    public LayerType Type { get; }

    /// <summary>Current activation values (post-sigmoid).</summary>
    public double[] Activations { get; }

    /// <summary>Pre-activation values (weighted sum + bias, before sigmoid).</summary>
    public double[] PreActivations { get; }

    /// <summary>Error deltas for backpropagation.</summary>
    public double[] Deltas { get; }

    /// <summary>Bias for each neuron.</summary>
    public double[] Biases { get; }

    /// <summary>Weights[j][i] = weight from input neuron i to this layer's neuron j.</summary>
    public double[][] Weights { get; }

    public Layer(int size, int inputSize, LayerType type, Random rng)
    {
        Size = size;
        InputSize = inputSize;
        Type = type;

        Activations = new double[size];
        PreActivations = new double[size];
        Deltas = new double[size];
        Biases = new double[size];

        // Xavier initialization for weights
        Weights = new double[size][];
        double scale = inputSize > 0 ? Math.Sqrt(2.0 / inputSize) : 0;

        for (int j = 0; j < size; j++)
        {
            Weights[j] = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                Weights[j][i] = (rng.NextDouble() * 2 - 1) * scale;
            }
            Biases[j] = (rng.NextDouble() * 2 - 1) * 0.1;
        }
    }
}
