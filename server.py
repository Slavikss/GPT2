
import os
import time
import requests
import json
import subprocess
import gdown
import torch
import tiktoken
import torch.nn as nn
from gpt2 import GPT
from torch.nn import functional as F
from train_gpt2 import GPTConfig
from collections import OrderedDict
from flask import Flask, request, jsonify
from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics
from prometheus_client import Counter

app = Flask(__name__, static_url_path="")
enc = tiktoken.get_encoding('gpt2')
metrics = GunicornInternalPrometheusMetrics(app)

# weights in dockerfile download to this dir
PATH_TO_WEIGHTS = 'weights/model_step8000'
PREDICTION_COUNT = Counter("predictions_total", "Number of predictions")
    

def download_file(url, output_directory='weights', output_file_name='model_step8000'):

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # If no output file name is provided, use the name from the URL
    if output_file_name is None:
        output_file_name = os.path.basename(url)

    # Construct the full path for the output file
    output_file = os.path.join(output_directory, output_file_name)

    if os.path.exists(output_file):
        print('weights are already cached')
    else:
        gdown.download(url, output_file)


def load(path_to_weights, device_type):
    model = GPT(GPTConfig(vocab_size = 50304))
    assert path_to_weights != None, "You must specify path to your model's weights"
    if device_type == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:   
        device = 'cpu'
    state_dict = torch.load(path_to_weights, map_location=torch.device(device))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # delete prefix "module."
        else:
            name = k
        new_state_dict[name] = v

    # loading weights
    model.load_state_dict(new_state_dict)
    model = torch.compile(model)
    model.to(device_type)
    model.eval()

    return model, device

model, device = 'model', 'cpu' #load(PATH_TO_WEIGHTS, 'cuda')

print(f'using {device}')

@app.route("/predict", methods=['POST'])
@metrics.gauge("api_in_progress", "requests in progress")
@metrics.counter("api_invocations_total", "number of invocations")
def generate(model=model, device=device, num_return_sequences=1, max_length=64):
    t0 = time.time()
    text = request.get_json(force=True)['message']
    # tokens = enc.encode(text)
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    # xgen = tokens.to(device)
    # while xgen.size(1) < max_length:
    #     with torch.no_grad():
    #         with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #             logits, _ = model(xgen)
    #         logits = logits[:, -1, :]
    #         probs = F.softmax(logits, dim=-1)
    #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    #         ix = torch.multinomial(topk_probs, 1)
    #         xcol = torch.gather(topk_indices, -1, ix)
    #         xgen = torch.cat((xgen, xcol), dim=1)

    # decoded_samples = []
    # for i in range(num_return_sequences):
    #     tokens = xgen[i, :max_length].tolist()
    #     decoded_samples.append(enc.decode(tokens))
    
    PREDICTION_COUNT.inc()

    t1 = time.time()

    return jsonify({
        "output": text,
        'gen_time': f'{round((t1-t0)*1000, 2)} ms'
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)

