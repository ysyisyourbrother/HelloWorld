import torch
import argparse
from medusa.model.medusa_model import MedusaModel

def main(args):
    model = MedusaModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = model.get_tokenizer()

    prompt ="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, 
    detailed, and polite answers to the user's questions. USER: hello ASSISTANT:"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
        model.base_model.device
    )
    output_stream = model.medusa_generate(
        input_ids,
        temperature=args.temperature,
        max_steps=args.max_steps,
    )

    # 从generator中获取输出结果
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="medusa-vicuna-7b-v1.3", help="Model name or path.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    args = parser.parse_args()
    main(args)