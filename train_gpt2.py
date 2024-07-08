import os
import time
import math
import torch
import tiktoken
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
# import wandb
from gpt2 import *

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class DDPSetup:
    def __init__(self):
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            assert torch.cuda.is_available()
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(1337)
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(1337)
            
class GPTTrainer:
    def __init__(self, config, ddp_setup):
        self.config = config
        self.ddp_setup = ddp_setup
        self.device = ddp_setup.device
        self.device_type = ddp_setup.device_type
        self.master_process = ddp_setup.master_process
        self.ddp = ddp_setup.ddp
        self.ddp_rank = ddp_setup.ddp_rank
        self.ddp_local_rank = ddp_setup.ddp_local_rank
        self.ddp_world_size = ddp_setup.ddp_world_size

        B, T = 16, 1024
        total_batch_size = 524288
        assert total_batch_size % (B * T * self.ddp_world_size) == 0
        self.grad_accum_steps = total_batch_size // (B * T * self.ddp_world_size)

        self.train_loader = DataLoaderLite(B, T, process_rank=self.ddp_rank, num_processes=self.ddp_world_size, split='train')
        self.val_loader = DataLoaderLite(B, T, process_rank=self.ddp_rank, num_processes=self.ddp_world_size, split='val')

        self.model = GPT(config)
        self.model.to(self.device)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        self.raw_model = self.model.module if self.ddp else self.model

        self.max_lr = 6e-4
        self.min_lr = 0.1 * self.max_lr
        self.warmup_steps = 715
        self.max_steps = 19073

        self.optimizer = self.raw_model.configure_optimizers(weight_decay=0.1, learning_rate=self.max_lr, device=self.device_type)
        
        if self.master_process:
            print(f'{total_batch_size=}')
            print(f'{self.grad_accum_steps=}')
            wandb.init(
                project="GPT2",
                config={
                    "learning_rate": self.max_lr,
                    "architecture": "GPT2",
                    "dataset": "FineWEB10B",
                    "epochs": 1,
                }
            )

    def get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_steps:
            return self.min_lr

        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def train(self):
        for step in range(self.max_steps):
            if step == self.max_steps - 1 or step % 4000 == 0:
                if self.master_process:
                    torch.save(self.model.state_dict(), f'model_step{step}')

            if (step >= 0 and step % 250 == 0):
                if self.master_process:
                    self.model.eval()
                    num_return_sequences = 4
                    max_length = 32
                    enc = tiktoken.get_encoding('gpt2')
                    tokens = enc.encode("Hello, I'm a language model,")
                    tokens = torch.tensor(tokens, dtype=torch.long)
                    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                    xgen = tokens.to(self.device)
                    while xgen.size(1) < max_length:
                        with torch.no_grad():
                            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                                logits, _ = self.model(xgen)
                            logits = logits[:, -1, :]
                            probs = F.softmax(logits, dim=-1)
                            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                            ix = torch.multinomial(topk_probs, 1)
                            xcol = torch.gather(topk_indices, -1, ix)
                            xgen = torch.cat((xgen, xcol), dim=1)
                    print()
                    print('#' * 10, f'validation_step_{step}', '#' * 10)
                    for i in range(num_return_sequences):
                        tokens = xgen[i, :max_length].tolist()
                        decoded = enc.decode(tokens)
                        print(f"sample {i}: {decoded}")
                    print('#' * 40)
                    print()

            if step % 100 == 0:
                self.model.eval()
                self.val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = self.val_loader.next_batch()
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                            _, loss = self.model(x, y)
                        loss /= val_loss_steps
                        val_loss_accum += loss.detach()

                if self.ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

                if self.master_process:
                    print(f'val_loss: {val_loss_accum.item():.4f}', end=' | ')
                    wandb.log({'val_loss': val_loss_accum.item(), 'log_val_loss': math.log(val_loss_accum.item())})

            t1 = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            loss_accum = 0.0

            for micro_step in range(self.grad_accum_steps):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)

                loss /= self.grad_accum_steps
                loss_accum += loss.detach()

                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)

                loss.backward()

            if self.ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.step()

            if self.device_type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.time()

            if self.master_process:
                print(f'step:{step}, loss:{loss_accum.item():.4f}, time:{round((t2-t1)*1000,4)}ms, norm:{norm:.4f}, lr: {lr:4f}'
                      f' tok/sec: {round((self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size) / (t2-t1),4)}')

                wandb.log({"step": step, "loss": loss_accum.item(), "norm": norm, 'lr': lr})

        if self.ddp:
            destroy_process_group()

if __name__ == '__main__':
    ddp_setup = DDPSetup()
    config = GPTConfig(vocab_size=50304)
    trainer = GPTTrainer(config, ddp_setup)
    trainer.train()
