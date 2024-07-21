import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from peer.dataset import PileDataset
from peer.model import PEERLanguageModel
from peer.trainer import train, validate
import matplotlib.pyplot as plt
import logging

def init_logging(local_rank):
    logging.basicConfig(level=logging.INFO if local_rank == 0 else logging.ERROR)

def plot_losses(train_losses, val_losses, epoch, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Epoch {epoch+1} Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}_losses.png'))
    plt.close()

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)

def load_data(tokenizer, batch_size):
    train_dataset = PileDataset('Salesforce/wikitext', tokenizer, split='train')
    val_dataset = PileDataset('Salesforce/wikitext', tokenizer, split='validation')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_sampler

def main():
    device = setup_distributed()
    local_rank = torch.distributed.get_rank()
    init_logging(local_rank)

    # Configurable parameters
    config = {
        "vocab_size": 50257,  # GPT-2 tokenizer vocab size
        "dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "num_experts": 512 * 512,
        "top_k": 16,
        "batch_size": 6,
        "num_epochs": 10,
        "learning_rate": 1e-4
    }

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = PEERLanguageModel(config['vocab_size'], config['dim'], config['num_layers'],
                              config['num_heads'], config['num_experts'], config['top_k']).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_loader, val_loader, train_sampler = load_data(tokenizer, config['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_sampler.set_epoch(epoch)
        logging.info(f"Epoch Training {epoch+1}/{config['num_epochs']}")
        train_loss, train_batch_losses = train(model, train_loader, optimizer, device)
        logging.info(f"Epoch Validation {epoch+1}/{config['num_epochs']}")
        val_loss, val_perplexity, val_batch_losses = validate(model, val_loader, device)
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")

        if local_rank == 0:
            plot_losses(train_batch_losses, val_batch_losses, epoch, 'plots')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_peer_language_model.pth')
            torch.save(model.state_dict(), 'final_peer_language_model.pth')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
