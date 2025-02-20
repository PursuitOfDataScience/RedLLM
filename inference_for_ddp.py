import torch
from transformers import AutoTokenizer
from pretrain_ddp import GPTConfig, GPT

def main():
    # Path to your saved model folder
    model_dir = "RedLLM"

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2) Load the model
    config = GPTConfig.from_pretrained("RedLLM")
    model = GPT.from_pretrained("RedLLM", config=config)
    model.to("cuda")
    # 3) Compute number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params} ({total_params/1e6:.2f} million)")

    # 4) Simple inference / text generation
    prompt = "花袭人有始有终，"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to("cuda")

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=500, 
            temperature=0.8,   
            top_k=50           
        )

    # 5) Decode and print
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()