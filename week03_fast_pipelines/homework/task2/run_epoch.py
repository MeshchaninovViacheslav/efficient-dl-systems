import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from transformer import TransformerModel, generate_square_subsequent_mask
from dataset import BrainDataset, BigBrainDataset, UltraBigBrainDataset, \
        UltraDuperBigBrainDataset, collate_fn, DataMode, UltraBigBrainBatchSampler


def get_gpt2_model() -> torch.nn.Module:
    # Initialize tokenizer to get vocabulary size
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer)
    
    # Model hyperparameters
    d_model = 1024  
    nhead = 8      
    d_hid = 4096   
    nlayers = 1    
    dropout = 0.1  
    
    # Create model using TransformerModel
    model = TransformerModel(
        ntoken=vocab_size,
        d_model=d_model,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout
    )
    
    return model


def run_epoch(data_mode: DataMode) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_gpt2_model().to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch_size = 256
    k = 640

    # Initialize dataset based on data mode
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path = 'wikitext-103-raw-v1', tokenizer=tokenizer)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path = 'wikitext-103-raw-v1', tokenizer=tokenizer)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        dataset = UltraBigBrainDataset(data_path = 'wikitext-103-raw-v1', tokenizer=tokenizer)
        batch_sampler = UltraBigBrainBatchSampler(dataset, batch_size=batch_size, k=k)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        dataset = UltraDuperBigBrainDataset(data_path = 'wikitext-103-raw-v1', tokenizer=tokenizer)
    else:
        raise NotImplementedError("Other data modes not implemented yet")
    
    if data_mode == DataMode.BRAIN or data_mode == DataMode.BIG_BRAIN:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=40, 
                                collate_fn=partial(collate_fn, data_mode=data_mode, tokenizer=tokenizer))
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=40,
                                collate_fn=partial(collate_fn, data_mode=data_mode, tokenizer=tokenizer))
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    
    
    # Warmup GPU
    print("Warming up GPU...")
    dummy_input = torch.randint(0, 1000, (640, batch_size)).to(device)
    dummy_mask = generate_square_subsequent_mask(640).to(device)
    for _ in range(10):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(dummy_input, dummy_mask)
    torch.cuda.synchronize()
    
    batch_times = []
    
    print(f"Running epoch with {data_mode.name} mode...")
    for batch in tqdm(dataloader, total=len(dataloader)):
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        else:
            input_ids = batch.to(device)
            attention_mask = generate_square_subsequent_mask(input_ids.size(0)).to(device)
        input_ids = input_ids.transpose(0, 1)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids, attention_mask)
        end_event.record()
        
        # Synchronize CUDA and record time
        torch.cuda.synchronize()
        batch_time = start_event.elapsed_time(end_event)
        batch_times.append(batch_time)
    
    # Calculate statistics
    batch_times = torch.tensor(batch_times)
    stats = {
        'min': batch_times.min().item(),
        'max': batch_times.max().item(),
        'mean': batch_times.mean().item(),
        'median': batch_times.median().item()
    }
    
    print(f"\nStatistics for {data_mode.name}:")
    print(f"Min batch time: {stats['min']:.2f} ms")
    print(f"Max batch time: {stats['max']:.2f} ms")
    print(f"Mean batch time: {stats['mean']:.2f} ms")
    print(f"Median batch time: {stats['median']:.2f} ms")


if __name__ == "__main__":
    run_epoch(DataMode.ULTRA_DUPER_BIG_BRAIN)
