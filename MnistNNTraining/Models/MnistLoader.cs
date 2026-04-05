namespace MnistNNTraining.Models;

/// <summary>
/// Loads MNIST data from the standard IDX binary format.
/// Expects files like: train-images-idx3-ubyte, train-labels-idx1-ubyte
/// Also supports raw byte arrays uploaded via the browser.
/// </summary>
public class MnistLoader
{
    /// <summary>
    /// A single MNIST sample: 28x28 pixel values normalized to [0,1] and a label 0-9.
    /// </summary>
    public record MnistSample(double[] Pixels, int Label, byte[] RawPixels);

    /// <summary>
    /// Parse MNIST images + labels from raw byte arrays (IDX format).
    /// </summary>
    public static List<MnistSample> Load(byte[] imageBytes, byte[] labelBytes)
    {
        var samples = new List<MnistSample>();

        // Parse image file header
        // Byte 0-3: magic number (2051)
        // Byte 4-7: number of images
        // Byte 8-11: rows (28)
        // Byte 12-15: cols (28)
        int imageMagic = ReadInt32BigEndian(imageBytes, 0);
        if (imageMagic != 2051)
            throw new FormatException($"Invalid image file magic number: {imageMagic} (expected 2051)");

        int numImages = ReadInt32BigEndian(imageBytes, 4);
        int rows = ReadInt32BigEndian(imageBytes, 8);
        int cols = ReadInt32BigEndian(imageBytes, 12);
        int pixelCount = rows * cols; // 784

        // Parse label file header
        // Byte 0-3: magic number (2049)
        // Byte 4-7: number of labels
        int labelMagic = ReadInt32BigEndian(labelBytes, 0);
        if (labelMagic != 2049)
            throw new FormatException($"Invalid label file magic number: {labelMagic} (expected 2049)");

        int numLabels = ReadInt32BigEndian(labelBytes, 4);
        int count = Math.Min(numImages, numLabels);

        for (int i = 0; i < count; i++)
        {
            int imageOffset = 16 + i * pixelCount;
            byte[] rawPixels = new byte[pixelCount];
            Array.Copy(imageBytes, imageOffset, rawPixels, 0, pixelCount);

            double[] pixels = new double[pixelCount];
            for (int p = 0; p < pixelCount; p++)
            {
                pixels[p] = rawPixels[p] / 255.0;
            }

            int label = labelBytes[8 + i];
            samples.Add(new MnistSample(pixels, label, rawPixels));
        }

        return samples;
    }

    /// <summary>
    /// Parse training data from the server's compact binary format.
    /// Format: [int32 count][for each: byte label, byte[784] pixels] (little-endian)
    /// </summary>
    public static List<MnistSample> LoadFromServerBinary(byte[] data)
    {
        var samples = new List<MnistSample>();

        if (data.Length < 4)
            throw new FormatException("Data too short — expected at least 4 bytes for count.");

        int count = BitConverter.ToInt32(data, 0); // little-endian
        int offset = 4;
        int recordSize = 1 + 784; // 1 byte label + 784 bytes pixels

        for (int i = 0; i < count; i++)
        {
            if (offset + recordSize > data.Length)
                break;

            byte label = data[offset];
            offset++;

            byte[] rawPixels = new byte[784];
            Array.Copy(data, offset, rawPixels, 0, 784);
            offset += 784;

            double[] pixels = new double[784];
            for (int p = 0; p < 784; p++)
            {
                pixels[p] = rawPixels[p] / 255.0;
            }

            samples.Add(new MnistSample(pixels, label, rawPixels));
        }

        return samples;
    }

    private static int ReadInt32BigEndian(byte[] data, int offset)
    {
        return (data[offset] << 24) |
               (data[offset + 1] << 16) |
               (data[offset + 2] << 8) |
               data[offset + 3];
    }
}
