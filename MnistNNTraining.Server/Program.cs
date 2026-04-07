var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

// CORS: allow the Blazor WASM client to call this API from a different origin
var allowedOrigins = builder.Configuration.GetSection("AllowedCorsOrigins").Get<string[]>();
if (allowedOrigins is { Length: > 0 })
{
    builder.Services.AddCors(options =>
        options.AddDefaultPolicy(policy =>
            policy.WithOrigins(allowedOrigins)
                  .AllowAnyHeader()
                  .AllowAnyMethod()));
}
else
{
    builder.Services.AddCors(options =>
        options.AddDefaultPolicy(policy =>
            policy.AllowAnyOrigin()
                  .AllowAnyHeader()
                  .AllowAnyMethod()));
}

var app = builder.Build();

app.UseCors();
app.UseStaticFiles();
app.UseBlazorFrameworkFiles();
app.UseRouting();

app.MapControllers();
app.MapFallbackToFile("index.html");

app.Run();