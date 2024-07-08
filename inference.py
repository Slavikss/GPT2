
import os
import requests
import subprocess
import gdown
import torch
import tiktoken
import torch.nn as nn
from gpt2 import GPT
from torch.nn import functional as F
from train_gpt2 import GPTConfig
from collections import OrderedDict

def download_file(url, output_directory='weights', output_file_name='model_step8000'):

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # If no output file name is provided, use the name from the URL
    if output_file_name is None:
        output_file_name = os.path.basename(url)

    # Construct the full path for the output file
    output_file = os.path.join(output_directory, output_file_name)

    gdown.download(url, output_file)


def load(path_to_weights, device_type):
    # download weights
    download_file('https://drive.google.com/uc?export=download&id=1DKDPW9x8EyFPa8O_uSw95t8dd3pVtAJI')

    model = GPT(GPTConfig(vocab_size = 50304))
    assert path_to_weights != None, "You must specify path to your model's weights"
    if device_type == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    state_dict = torch.load(path_to_weights)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # delete prefix "module."
        else:
            name = k
        new_state_dict[name] = v

    # loading weights
    model.load_state_dict(new_state_dict)
    model.to(device_type)
    model.eval()

    return model, device

def generate(model, device, text, num_return_sequences, max_length):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    decoded_samples = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded_samples.append(enc.decode(tokens))

    return decoded_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", type=str, default="None", help="path to weights of your own model")
    parser.add_argument("-text", "--text", type=str, default="None", help="Context to model")
    parser.add_argument("-num_return_seq", "--num_return_seq", type=int, default= 1, help="How many samples return")
    parser.add_argument("-max_length", "--max_length", type=int, default=64, help="Length of sample")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()

    model, device = load(args.path, args.device)    
    generated_samples = generate(model, device, args.text, args.num_return_seq, args.max_length)


