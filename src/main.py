"""
TODO
"""

import torch
import yaml
from contextlib import nullcontext
import numpy as np
import os
from argparse import ArgumentParser

from models import BigramLanguageModel


def main(params, cont=False):
    # =================================
    # ===== Read in training data =====
    # =================================
    with open('mb_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # get all character appearing in the text, optionally add all integers
    chars = sorted(
        list(set(text))
        # + [str(k) for k in range(10)]
    )
    # create a mapping from characters to integers
    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in enumerate(chars)}

    # ==========================================
    # ===== Load parameters from json file =====
    # ==========================================
    n_embed = params['n_embed']
    learn_rate = params['learn_rate']
    eval_iters = params['eval_iters']
    iter_num = params['iter_num']
    n_layers = params['n_layers']
    num_heads = params['num_heads']
    block_size = params['block_size']
    batch_size = params['batch_size']
    output_size = params['output_size']
    dropout = params['dropout']
    device = params['device']
    dtype = params['dtype']
    seed = params['seed']

    torch.manual_seed(seed)
    vocab_size = len(chars)

    # Create folder and file to save model after training
    save_file = 'save/model.pt'
    save_path = os.path.dirname(save_file)
    if cont:
        assert os.path.exists(save_file)
        print('Continuing from already trained model')
    else:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            with open(save_file, 'w'):
                pass

    # ===========================
    # ===== Device settings =====
    # ===========================
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # for later use in torch.autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    # ==================================
    # ===== Prepare data for usage =====
    # ==================================
    # let's now encode the entire text dataset and store it into a torch.Tensor
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # Let's now split up the data into train and validation sets
    train = 0.8  # 80% of the data is for training, the rest for evaluation
    n = int(train*len(data))
    train_data = data[:n]
    val_data = data[n:]

    xb, yb = get_batch('train', block_size, batch_size,
                       train_data, val_data, device, device_type)

    # ============================
    # ===== Define the model =====
    # ============================
    model = BigramLanguageModel(vocab_size, block_size,
                                n_embed, num_heads, n_layers,
                                dropout)
    if cont:
        model.load_state_dict(torch.load(save_file))

    model.eval()
    model.to(device)
    _, loss = model(xb, yb)
    # Compute initial loss, should be -ln(1/vocab_size)
    if cont:
        print(f'Initial loss: {loss.item()}')
    else:
        print(f'Initial loss: {loss.item()} \
                , compare with perfectly flat prior: {-np.log(1/vocab_size)}')

    # print initial output -> totally random, or even worse
    # print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    # ===========================
    # ===== Train the model =====
    # ===========================

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # do actual training
    for _ in range(iter_num):

        # sample a batch of data
        xb, yb = get_batch('train', block_size, batch_size,
                           train_data, val_data, device, device_type)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the state of the model
    torch.save(model.state_dict(), save_file)

    # =================================================
    # ==== Print output resulting from a new line =====
    # =================================================
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=output_size,
          block_size=block_size)[0].tolist(), itos))
    losses = estimate_loss(model, eval_iters, block_size,
                           batch_size, train_data, val_data,
                           device, device_type, ctx)
    print(f"\nstep {iter_num}: \
            train loss {losses['train']:.4f}, \
            val loss {losses['val']:.4f}")


def encode(s: str, stoi: dict):
    """
    encoder: take a string, output a list of integers
    """
    return [stoi[c] for c in s]


def decode(l: list, itos: dict):
    """
    decoder: take a list of integers, output a string
    """
    return ''.join([itos[i] for i in l])


def get_batch(split, block_size, batch_size, train_data, val_data, device, device_type):
    """
    Get a batch (consisting of chunks) of the data.

    Parameters
    ----------
    split : str
        'train' or else, if training mode or not

    TODO
    """
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, train_data, val_data, device, device_type, ctx):
    """ helps estimate an arbitrarily accurate loss over either split using many batches

    Parameters
    ----------
    TODO
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size,
                             train_data, val_data, device, device_type)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--cont',
                        action='store_true',
                        help="If the training should be continued from an existing file.")
    args = parser.parse_args()
    cont = args.cont

    # run main with default params file
    filepath = 'params.yml'
    with open(filepath) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    main(params, cont)
