using Microsoft.AspNetCore.Mvc;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MnistNNTraining.Server.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DataController : ControllerBase
{
    private readonly IWebHostEnvironment _env;
    private readonly ILogger<DataController> _logger;

    public DataController(IWebHostEnvironment env, ILogger<DataController> logger)
    {
        _env = env;
        _logger = logger;
    }

    /// <summary>
    /// Returns training data info (how many images per digit folder).
    /// </summary>
    [HttpGet("info")]
    public IActionResult GetInfo()
    {
        var dataPath = GetTrainingDataPath();
        if (!Directory.Exists(dataPath))
            return NotFound(new { error = "TrainingData folder not found. Create a TrainingData/ folder with subfolders 0-9." });

        var info = new Dictionary<string, int>();
        int total = 0;
        for (int digit = 0; digit <= 9; digit++)
        {
            var folder = Path.Combine(dataPath, digit.ToString());
            int count = 0;
            if (Directory.Exists(folder))
            {
                count = Directory.GetFiles(folder, "*.jpg").Length
                      + Directory.GetFiles(folder, "*.jpeg").Length
                      + Directory.GetFiles(folder, "*.png").Length;
            }
            info[digit.ToString()] = count;
            total += count;
        }

        return Ok(new { total, perDigit = info, path = dataPath });
    }

    /// <summary>
    /// Loads all training images and returns them as a compact binary payload.
    /// Format: [int32 count][for each sample: byte label, byte[784] pixels]
    /// All integers are little-endian.
    /// </summary>
    [HttpGet("training")]
    public IActionResult GetTrainingData([FromQuery] int maxPerDigit = 0)
    {
        var dataPath = GetTrainingDataPath();
        if (!Directory.Exists(dataPath))
            return NotFound(new { error = "TrainingData folder not found." });

        var samples = new List<(byte label, byte[] pixels)>();

        for (int digit = 0; digit <= 9; digit++)
        {
            var folder = Path.Combine(dataPath, digit.ToString());
            if (!Directory.Exists(folder)) continue;

            var files = Directory.GetFiles(folder, "*.*")
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f)
                .ToList();

            if (maxPerDigit > 0)
                files = files.Take(maxPerDigit).ToList();

            foreach (var file in files)
            {
                try
                {
                    byte[] pixels = LoadImageAsGrayscale(file);
                    samples.Add(((byte)digit, pixels));
                }
                catch (Exception ex)
                {
                    _logger.LogWarning("Failed to load {File}: {Error}", file, ex.Message);
                }
            }
        }

        _logger.LogInformation("Loaded {Count} training samples from {Path}", samples.Count, dataPath);

        // Build binary payload: [count:int32][label:byte, pixels:byte[784]] × count
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(samples.Count); // int32
        foreach (var (label, pixels) in samples)
        {
            writer.Write(label);     // 1 byte
            writer.Write(pixels);    // 784 bytes
        }

        return File(ms.ToArray(), "application/octet-stream");
    }

    /// <summary>
    /// Load a single image, resize to 28x28 if needed, convert to grayscale.
    /// Returns 784 bytes (row-major, 0=black, 255=white).
    /// </summary>
    private static byte[] LoadImageAsGrayscale(string path)
    {
        using var image = Image.Load<Rgba32>(path);

        // Resize to 28x28 if not already
        if (image.Width != 28 || image.Height != 28)
        {
            image.Mutate(ctx => ctx.Resize(28, 28));
        }

        byte[] pixels = new byte[784];
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                var pixel = image[x, y];
                // Convert to grayscale using luminance formula
                byte gray = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                pixels[y * 28 + x] = gray;
            }
        }

        return pixels;
    }

    private string GetTrainingDataPath()
    {
        // Look for TrainingData folder relative to the solution root
        var contentRoot = _env.ContentRootPath;
        var solutionDir = Path.GetFullPath(Path.Combine(contentRoot, ".."));
        return Path.Combine(solutionDir, "TrainingData");
    }
}
