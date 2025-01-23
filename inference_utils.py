def test_model(model, tokenizer, input_text):
    messages = [
        {"role": "user", "content": input_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True,
                             temperature=1.5, min_p=0.1)
    return tokenizer.batch_decode(outputs)[0]