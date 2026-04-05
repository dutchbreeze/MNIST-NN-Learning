using MnistNNTraining.Models;

namespace MnistNNTraining.Services;

/// <summary>
/// Manages the training loop, play/pause, step control, and speed.
/// </summary>
public class TrainingService
{
    public NeuralNetwork Network { get; private set; } = new();
    public List<MnistLoader.MnistSample> TrainingData { get; private set; } = new();
    public bool IsDataLoaded => TrainingData.Count > 0;

    // Training state
    public bool IsTraining { get; private set; }
    public bool IsPaused { get; private set; } = true;
    public int CurrentSampleIndex { get; private set; }
    public MnistLoader.MnistSample? CurrentSample { get; private set; }
    public int BatchSize { get; set; } = 1;
    public int StepsPerTick { get; set; } = 1;

    // Stats
    public double RunningLoss { get; private set; }
    public int RunningCount { get; private set; }
    public double AverageLoss => RunningCount > 0 ? RunningLoss / RunningCount : 0;
    public List<double> LossHistory { get; } = new();
    public List<double> AccuracyHistory { get; } = new();

    // Prediction tracking (correct vs incorrect over time)
    public List<int> CorrectHistory { get; } = new();
    public List<int> IncorrectHistory { get; } = new();
    private int _snapshotCorrect;
    private int _snapshotIncorrect;

    // Snapshot-level loss/accuracy tracking
    private double _snapshotLoss;
    private int _snapshotLossCount;
    private int _snapshotAccuracyCorrect;
    private int _snapshotAccuracyCount;

    // Periodic snapshot interval (every N samples processed, push a snapshot)
    public int SnapshotInterval { get; set; } = 100;
    private int _samplesSinceSnapshot;

    // Total tick counter
    public long TotalTicks { get; private set; }

    private readonly Random _shuffleRng = new(42);
    private int[] _shuffledIndices = Array.Empty<int>();

    public event Action? OnStateChanged;

    public void LoadData(byte[] imageBytes, byte[] labelBytes)
    {
        TrainingData = MnistLoader.Load(imageBytes, labelBytes);
        ResetShuffleOrder();
        CurrentSampleIndex = 0;
        OnStateChanged?.Invoke();
    }

    public void LoadFromServerBinary(byte[] data)
    {
        TrainingData = MnistLoader.LoadFromServerBinary(data);
        ResetShuffleOrder();
        CurrentSampleIndex = 0;
        OnStateChanged?.Invoke();
    }

    public void ResetNetwork()
    {
        Network.Reset();
        CurrentSampleIndex = 0;
        RunningLoss = 0;
        RunningCount = 0;
        TotalTicks = 0;
        _samplesSinceSnapshot = 0;
        _snapshotCorrect = 0;
        _snapshotIncorrect = 0;
        _snapshotLoss = 0;
        _snapshotLossCount = 0;
        _snapshotAccuracyCorrect = 0;
        _snapshotAccuracyCount = 0;
        LossHistory.Clear();
        AccuracyHistory.Clear();
        CorrectHistory.Clear();
        IncorrectHistory.Clear();
        IsPaused = true;
        IsTraining = false;
        ResetShuffleOrder();
        OnStateChanged?.Invoke();
    }

    public void Play()
    {
        if (!IsDataLoaded) return;
        IsPaused = false;
        IsTraining = true;
        OnStateChanged?.Invoke();
    }

    public void Pause()
    {
        IsPaused = true;
        OnStateChanged?.Invoke();
    }

    public void TogglePlayPause()
    {
        if (IsPaused) Play(); else Pause();
    }

    /// <summary>
    /// Execute a single training tick (processes StepsPerTick samples).
    /// Called by the UI timer.
    /// </summary>
    public void Tick()
    {
        if (IsPaused || !IsDataLoaded) return;

        TotalTicks++;

        for (int s = 0; s < StepsPerTick; s++)
        {
            if (CurrentSampleIndex >= TrainingData.Count)
            {
                // End of epoch
                Network.EndEpoch();

                // Flush any remaining snapshot data at epoch boundary
                if (_samplesSinceSnapshot > 0)
                {
                    PushSnapshot();
                }

                RunningLoss = 0;
                RunningCount = 0;
                CurrentSampleIndex = 0;
                ResetShuffleOrder();
            }

            int idx = _shuffledIndices[CurrentSampleIndex];
            CurrentSample = TrainingData[idx];
            double loss = Network.TrainSingle(CurrentSample.Pixels, CurrentSample.Label);

            // Track correct/incorrect for prediction chart
            int predicted = Array.IndexOf(Network.Layers[^1].Activations, Network.Layers[^1].Activations.Max());
            if (predicted == CurrentSample.Label)
            {
                _snapshotCorrect++;
                _snapshotAccuracyCorrect++;
            }
            else
            {
                _snapshotIncorrect++;
            }

            // Track loss for loss chart
            _snapshotLoss += loss;
            _snapshotLossCount++;
            _snapshotAccuracyCount++;

            RunningLoss += loss;
            RunningCount++;
            CurrentSampleIndex++;
            _samplesSinceSnapshot++;

            // Push a snapshot every SnapshotInterval samples
            if (_samplesSinceSnapshot >= SnapshotInterval)
            {
                PushSnapshot();
            }
        }

        OnStateChanged?.Invoke();
    }

    /// <summary>
    /// Push accumulated snapshot data to all history lists.
    /// </summary>
    private void PushSnapshot()
    {
        // Prediction chart data
        CorrectHistory.Add(_snapshotCorrect);
        IncorrectHistory.Add(_snapshotIncorrect);
        _snapshotCorrect = 0;
        _snapshotIncorrect = 0;

        // Loss/accuracy chart data
        double avgLoss = _snapshotLossCount > 0 ? _snapshotLoss / _snapshotLossCount : 0;
        double accuracy = _snapshotAccuracyCount > 0 ? (double)_snapshotAccuracyCorrect / _snapshotAccuracyCount * 100.0 : 0;
        LossHistory.Add(avgLoss);
        AccuracyHistory.Add(accuracy);

        _snapshotLoss = 0;
        _snapshotLossCount = 0;
        _snapshotAccuracyCorrect = 0;
        _snapshotAccuracyCount = 0;
        _samplesSinceSnapshot = 0;
    }

    /// <summary>
    /// Perform a single forward pass (no training) for visualization.
    /// </summary>
    public void StepForwardOnly()
    {
        if (!IsDataLoaded) return;

        int idx = _shuffledIndices[CurrentSampleIndex % TrainingData.Count];
        CurrentSample = TrainingData[idx];
        Network.Forward(CurrentSample.Pixels);
        OnStateChanged?.Invoke();
    }

    /// <summary>
    /// Perform a single training step.
    /// </summary>
    public void StepTrain()
    {
        if (!IsDataLoaded) return;

        if (CurrentSampleIndex >= TrainingData.Count)
        {
            Network.EndEpoch();
            if (_samplesSinceSnapshot > 0)
            {
                PushSnapshot();
            }
            RunningLoss = 0;
            RunningCount = 0;
            CurrentSampleIndex = 0;
            ResetShuffleOrder();
        }

        int idx = _shuffledIndices[CurrentSampleIndex];
        CurrentSample = TrainingData[idx];
        double loss = Network.TrainSingle(CurrentSample.Pixels, CurrentSample.Label);
        RunningLoss += loss;
        RunningCount++;
        CurrentSampleIndex++;

        OnStateChanged?.Invoke();
    }

    /// <summary>
    /// Returns model size info for display.
    /// </summary>
    public ModelSizeInfo GetModelSizeInfo()
    {
        int totalWeights = 0;
        int totalBiases = 0;

        for (int l = 1; l < Network.Layers.Count; l++)
        {
            var layer = Network.Layers[l];
            totalBiases += layer.Size;
            totalWeights += layer.Size * layer.InputSize;
        }

        int totalParams = totalWeights + totalBiases;
        // Each param is a double (8 bytes)
        double sizeKB = totalParams * 8.0 / 1024.0;

        return new ModelSizeInfo
        {
            TotalWeights = totalWeights,
            TotalBiases = totalBiases,
            TotalParameters = totalParams,
            SizeKB = sizeKB,
            Layers = Network.Layers.Select(l => $"{l.Size}").ToArray()
        };
    }

    private void ResetShuffleOrder()
    {
        _shuffledIndices = Enumerable.Range(0, TrainingData.Count).ToArray();
        // Fisher-Yates shuffle
        for (int i = _shuffledIndices.Length - 1; i > 0; i--)
        {
            int j = _shuffleRng.Next(i + 1);
            (_shuffledIndices[i], _shuffledIndices[j]) = (_shuffledIndices[j], _shuffledIndices[i]);
        }
    }
}

public class ModelSizeInfo
{
    public int TotalWeights { get; set; }
    public int TotalBiases { get; set; }
    public int TotalParameters { get; set; }
    public double SizeKB { get; set; }
    public string[] Layers { get; set; } = Array.Empty<string>();
}