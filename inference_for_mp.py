import torch
from transformers import AutoTokenizer
from mp_pretrain import GPTConfig, GPTModelParallel

def main():
    print("Found", torch.cuda.device_count(), "CUDA devices.")

    model_dir = "RedLLM_MP"

    # 1) Load config & tokenizer
    config = GPTConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2) Build pipeline‐parallel model from checkpoint
    model = GPTModelParallel.from_pretrained(model_dir, config=config)

    # --------------------------------------------
    # 3) Manually fix device placement
    # --------------------------------------------

    # a) token_embedding is an nn.Embedding module, so we can just .to(...)
    model.token_embedding.to(model.devices[0])
    
    # b) position_embedding is a Parameter, so we move .data
    model.position_embedding.data = model.position_embedding.data.to(model.devices[0])

    # c) drop is an nn.Dropout, so .to(...)
    model.drop.to(model.devices[0])

    # d) pipeline_stages is a ModuleList of stages
    for i, stage in enumerate(model.pipeline_stages):
        stage.to(model.devices[i])

    # e) final LN + head on the last GPU
    model.ln_f.to(model.devices[-1])
    model.head.to(model.devices[-1])

    model.eval()

    # 4) Prepare input on GPU 0
    prompt = "花袭人有始有终，"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.devices[0])

    # 5) Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=500,
            temperature=0.8,
            top_k=50
        )

    # 6) Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
