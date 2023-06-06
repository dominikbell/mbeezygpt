"""
TODO
"""

import os
import pickle
import yaml
import torch
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from time import time
from contextlib import nullcontext

from models import BigramLanguageModel


def main():
    # ================================
    # ===== Main Argument Parser =====
    # ================================
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='available commands',
                                       metavar='COMMAND',
                                       dest='command')

    # ====================================
    # ===== Training Argument Parser =====
    # ====================================
    parser_train = subparsers.add_parser('train',
                                         formatter_class=lambda prog: RawTextHelpFormatter(
                                             prog, max_help_position=40),
                                         help='train the model',
                                         description='Train the model.')

    parser_train.add_argument('inputfile',
                              type=str,
                              nargs='?',
                              help='On which data to train the model.',
                              metavar='inputfile',
                              default='mb_input.txt')

    parser_train.add_argument('--cont',
                              action='store_true',
                              help="If the training should be continued from an existing file.")

    # =====================================
    # ===== Freestyle Argument Parser =====
    # =====================================
    parser_freestyle = subparsers.add_parser('freestyle',
                                             formatter_class=lambda prog: RawTextHelpFormatter(
                                                 prog, max_help_position=40),
                                             help='spit some bars',
                                             description='Spit some bars.')

    parser_freestyle.add_argument('-o', '--output_size',
                                  type=int,
                                  nargs=1,
                                  help='Length of the desired output (in tokens).',
                                  metavar='output_size',
                                  default=[100])

    parser_freestyle.add_argument('prompt',
                                  type=str,
                                  nargs='?',
                                  help='Prompt for a freestyle (passed as a string)',
                                  metavar='prompt',
                                  default='\n')

    # ================================
    # ===== Parse all Arguments =====
    # ================================
    args = parser.parse_args()
    command = args.command

    # Exit if no command is given
    if command is None:
        print('No command given, exiting...')
        exit()

    # Run main with default params file
    filepath = 'params.yml'
    with open(filepath) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Read out arguments or set defaults
    cont = args.cont if 'cont' in args else False
    output_size = args.output_size[0] if 'output_size' in args else params['output_size']

    # Define path for saving and load for freestyle
    save_path = 'save/'
    save_file = os.path.join(save_path, 'model.pt')
    if command == 'freestyle':
        assert os.path.exists(save_file), \
            "The model must have been trained and saved before being able to freestyle!"
        prompt = args.prompt

    # ================================================
    # ===== Read in saved model or training data =====
    # ================================================
    if command == 'train':
        inputfile = args.inputfile
        with open(inputfile, 'r', encoding='utf-8') as f:
            text = f.read()
        n = 2

    # load tokens and dicts in case of continuing
    if cont or command == 'freestyle':
        tokens, itot, ttoi = load_tokens_from_file(save_path)
        n = len(tokens[0])
    else:
        tokens, itot, ttoi = get_tokens_from_text(text, n=n)
        save_tokens_to_file(tokens, itot, ttoi, save_path)

    # Define the Model
    model_param_keys = ['block_size',
                        'n_embed', 'n_heads', 'n_layers', 'dropout']
    model_params = {x: params[x] for x in model_param_keys}
    model_params['vocab_size'] = len(tokens)
    model = BigramLanguageModel(**model_params)

    if cont or command == 'freestyle':
        model.load_state_dict(torch.load(save_file))

    # ============================
    # ===== Run the Training =====
    # ============================
    if command == 'train':
        start_time = time()
        train(model, params, save_path, text, tokens, ttoi, n=n, cont=cont)
        end_time = time()
        print(f'This took {np.round(end_time - start_time, 3)} seconds.')

    # Give some output
    bars = freestyle(model, itot, ttoi,
                     model_params['block_size'],
                     n=n,
                     output_size=output_size,
                     prompt=prompt)
    print(bars)


def train(model, params, save_path, text, tokens, ttoi, n=1, cont=False):
    """ Train the model, give some output and print the loss

    Parameters
    ----------
    TODO
    """
    # ==========================================
    # ===== Load parameters from json file =====
    # ==========================================
    learn_rate = params['learn_rate']
    eval_iters = params['eval_iters']
    iter_num = params['iter_num']
    block_size = params['block_size']
    batch_size = params['batch_size']
    device = params['device']
    dtype = params['dtype']
    seed = params['seed']

    torch.manual_seed(seed)
    vocab_size = len(tokens)

    # Create folder and file to save model after training
    save_file = os.path.join(save_path, 'model.pt')
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
    data = torch.tensor(encode(text, ttoi, n=n), dtype=torch.long)

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
    model.eval()
    model.to(device)
    _, loss = model(xb, yb)
    # Compute initial loss, should be -ln(1/vocab_size)
    if cont:
        print(f'Initial loss: {loss.item()}')
    else:
        print(
            f'Initial loss: {loss.item()} \tcompare with perfectly flat prior: {-np.log(1/vocab_size)}')

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
    losses = estimate_loss(model, eval_iters, block_size,
                           batch_size, train_data, val_data,
                           device, device_type, ctx)
    print(f"\nAfter {iter_num} steps: \
            train loss {losses['train']:.4f}, \
            val loss {losses['val']:.4f}")


def freestyle(model, itot, ttoi, block_size, n=1, prompt=None, output_size=100, device='cpu'):
    """ Given a prompt, spit some lines.

    Parameters
    ----------
    model : torch.nn.Module
        the model for the text generation

    itot : dict
        integers to tokens dictionary

    prompt : str
        input to the model as basis for the freestyle. Default is new line character

    output_size : int
        How many tokens to output

    device : str
        'cpu' or 'cuda' or 'mps'
    """
    if prompt == '\n':
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.tensor(
            np.atleast_2d(np.array(encode(prompt, ttoi, n=n))),
            dtype=torch.long, device=device)
    return decode(model.generate(context, max_new_tokens=output_size,
                                 block_size=block_size)[0].tolist(), itot)


def get_tokens_from_text(text, n=1):

    if n == 1:
        tokens = sorted(list(set(text)))
    elif n == 2:
        tokens = sorted(
            list(set([a + b for a, b in zip(text[:-1], text[1:])])))
    else:
        raise NotImplementedError

    # create a mapping from tokens to integers
    itot = {i: t for i, t in enumerate(tokens)}
    ttoi = {t: i for i, t in enumerate(tokens)}

    return tokens, itot, ttoi


def save_tokens_to_file(tokens, itot, ttoi, filepath):
    """ saves the given tokens to the specified file

    Parameters
    ----------
    TODO
    """
    with open(os.path.join(filepath, 'tokens'), 'wb') as file:
        pickle.dump(tokens, file)

    with open(os.path.join(filepath, 'itot'), 'wb') as file:
        pickle.dump(itot, file)

    with open(os.path.join(filepath, 'ttoi'), 'wb') as file:
        pickle.dump(ttoi, file)


def load_tokens_from_file(filepath):
    """ saves the given tokens to the specified file

    Parameters
    ----------
    TODO
    """
    with open(os.path.join(filepath, 'tokens'), 'rb') as file:
        tokens = pickle.load(file)

    with open(os.path.join(filepath, 'itot'), 'rb') as file:
        itot = pickle.load(file)

    with open(os.path.join(filepath, 'ttoi'), 'rb') as file:
        ttoi = pickle.load(file)

    return tokens, itot, ttoi


def encode(s: str, ttoi: dict, n=1):
    """ encoder: take a string, output a list of integers
    """
    if n == 1:
        res = [ttoi[c] for c in s]
    elif n == 2:
        res = [ttoi[c] for c in [a+b for a, b in zip(s[::2], s[1::2])]]
    else:
        raise NotImplementedError

    return res


def decode(l: list, itot: dict):
    """ decoder: take a list of integers, output a string
    """
    res = ''.join([itot[i] for i in l])

    return res


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
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
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
    main()
