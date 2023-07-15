from langchain.llms import OpenAI
import json
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from gpt import GPT, GPTRewardModel, HFGPTRewardModel
from configs import get_configs
from tqdm import tqdm
import torch
import tiktoken
import click
import json
import csv
import openai
import numpy as np

def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode


def generate_gpt2(model, prompt, device):
    model.eval()
    model.to(device)
    max_new_tokens = 100
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    y = model.generate(x,
                       max_new_tokens,
                       temperature=temperature,
                       top_k=top_k)

    res = decode(y[0].cpu().tolist())
    end = res.find("<|endoftext|>")
    if end > 0:
        return res[:end]
    else:
        return res


@click.command()
@click.option('--sft', '-s')
@click.option('--ppo', '-p')
def main(sft, ppo):
    #keys = json.load(open("openai.key"))
    os.environ["OPENAI_API_KEY"] = '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'#keys["OPENAI_API_KEY"]
    #os.environ["http_proxy"] = "127.0.0.1:7890"
    #os.environ["https_proxy"] = "127.0.0.1:7890"
    proxies = {'http': "http://127.0.0.1:7890",'https': "http://127.0.0.1:7890"}
    openai.proxy = proxies
    prompts = np.load('./gpt2_safety_alignment/gpt2_eval.npy')
    '''
    with open("prompts.csv") as fp:
        reader = csv.DictReader(fp)
        prompts = [row["prompt"] for row in reader]
    '''

    print("Run inference")
    if os.path.exists("responses.json"):
        with open("responses.json") as fp:
            responses = json.load(fp)
    else:
        device = "cuda"
        cfg = get_configs("gpt2-medium/dropout")
        with torch.inference_mode():
            gpt_vanilla = GPT.from_pretrained(cfg)
            gpt_sft = GPT.from_checkpoint(
                cfg,
                sft)
            gpt_ppo = GPT.from_checkpoint(
                cfg,
                ppo)

            responses = []
            for prompt in tqdm(prompts):
                responses.append({
                    "vanilla": generate_gpt2(gpt_vanilla, f"Human: {prompt}\n\nAssistant: ", device)[
                               len(f"Human: {prompt}\n\nAssistant: "):],
                    "sft": generate_gpt2(gpt_sft, f"Human: {prompt}\n\nAssistant: ", device)[
                           len(f"Human: {prompt}\n\nAssistant: "):],
                    "ppo": generate_gpt2(gpt_ppo, f"Human: {prompt}\n\nAssistant: ", device)[
                           len(f"Human: {prompt}\n\nAssistant: "):],
                    "prompt": prompt
                })
            with open("responses_sft2000_ppo1.json", "w") as fp:
                json.dump(responses, fp)


if __name__ == '__main__':
    main()
