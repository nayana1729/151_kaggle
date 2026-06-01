import json
import csv
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MAX_NEW_TOKENS = 4096


def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        system = ("You are an expert mathematician. Select the best answer. "
                  "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}.")
        return system, f"{question}\n\nOptions:\n{opts}"
    system = ("You are an expert mathematician. Solve step by step. You MUST end "
              "your response with the final answer in \\boxed{}. Be concise. "
              "Multiple answers: \\boxed{3, 7}.")
    return system, question


def run_inference(private_path, output_csv, checkpoint="checkpoint.json"):
    done = {}
    if os.path.exists(checkpoint):
        with open(checkpoint) as f:
            done = json.load(f)
        print(f"Resuming: {len(done)} done")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
                                              padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")

    data = [json.loads(l) for l in open(private_path)]
    print(f"Total: {len(data)}")

    for i, item in enumerate(data):
        qid = str(item['id'])
        if qid in done:
            continue
        system, user = build_prompt(item['question'], item.get('options'))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=0.6, top_p=0.95, do_sample=True)
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        done[qid] = response
        with open(checkpoint, 'w') as f:
            json.dump(done, f)
        print(f"[{i+1}/{len(data)}] {qid}: {response[:80]}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['id', 'response'])
        for item in data:
            resp = done.get(str(item['id']), '')
            if not resp.strip():
                resp = 'No answer. \\boxed{0}'
            writer.writerow([item['id'], resp])
    print(f"Done! Saved {output_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private-path", default="data/private.jsonl")
    ap.add_argument("--output-csv", default="submission.csv")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Accepted for harness compatibility; unused (single GPU).")
    args = ap.parse_args()
    run_inference(args.private_path, args.output_csv)


if __name__ == "__main__":
    main()
