import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.float32,
)
tokenizer = pipeline.tokenizer

context = torch.randn(1, 64)  # fake time series
breakpoint()
# manually step through the tokenizer
token_ids, attention_mask, scale = tokenizer.context_input_transform(context)

centers = tokenizer.centers
indices = torch.clamp(token_ids - pipeline.model.config.n_special_tokens, 0, len(centers) - 1)
quantized = centers[indices]

print("original:  ", context[0, :8])
print("scale:     ", scale)
print("token_ids: ", token_ids[0, :8])
print("quantized: ", quantized[0, :8])
# run the full prediction to trigger output_transform
forecast = pipeline.predict(context, prediction_length=12)

print("forecast shape:", forecast.shape)  # (1, num_samples, 12)
print("forecast sample:", forecast[0, 0, :])  # first sample path