using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using MnistNNTraining;
using MnistNNTraining.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

var apiBase = builder.Configuration["ApiBaseUrl"];
var baseAddress = string.IsNullOrEmpty(apiBase)
    ? builder.HostEnvironment.BaseAddress
    : apiBase;

builder.Services.AddScoped(sp => new HttpClient
{
    BaseAddress = new Uri(baseAddress),
    Timeout = TimeSpan.FromMinutes(5)
});
builder.Services.AddSingleton<TrainingService>();

await builder.Build().RunAsync();