## GPT2 efficient implementation w/production-ready deploy

Run `docker-compose up --build` to start the server

**Endpoints**: \
Server API: `localhost:8081/predict`. Accepts `POST queries with -d {'message': 'text'}`     
Prometheus metrics: `localhost:9090/metrics` \
Prometheus UI: `localhost:9090` \
Grafana: `localhost:3030`

Alternatively, you can run scripts:
- `pyton3 fineweb.py` for dataset preparation
- `python3 train_gpt2.py` for model training from scratch
- `python3 hellaswag.py` to run [*Hellaswag*](https://huggingface.co/datasets/Rowan/hellaswag) scoring the model

\
Model description:
- `GPT2 124M`, `fp16`
- `block_size: 1024, vocab_size: 50304, n_layer: 12, n_head: 12, n_embd: 768`
- Trained on [FineWeb10B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) sample with 2x RTX4090 24GB
- Gradient accumulation for 0.5M batchsize
- Optimization: Flash attention, weights initialization, *"beautiful"* numbers - powers of 2, torch.compile(), torch.autocast() etc
  
TODO:
- make ONNX model