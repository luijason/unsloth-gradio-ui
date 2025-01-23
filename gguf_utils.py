def convert_to_gguf(model, tokenizer, output_path, quantization_method="q8_0"):
    try:
        model.save_pretrained_gguf(output_path, tokenizer, quantization_method=quantization_method)
        return f"Model successfully converted to GGUF format: {output_path}-unsloth-{quantization_method}.gguf"
    except Exception as e:
        return f"Error converting to GGUF: {str(e)}"