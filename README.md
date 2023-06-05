# mbeezygpt

This beautiful basic transformer model aims at learning the lyrics of the grand austrian master MoneyBoy. The model was initially constructed following a [tutorial by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY); go check it out and leave him a like!

# How to install

Run
```
python3 -m pip [-e] .
```
to install into your machine. The option `[-e]` is for developer mode, which uses the files in this directory directly to run the program.

# How to train & run the model

The model can be trained by running
```
mbeezy train
```
The default training dataset is `mb_input.txt`. After the training, the state of the network is saved under `save/model.pt`; if you want to resume training from this saved state pass the flag `--cont`.

Default parameters are stored in `params.yml` and can be changed there.

Output can be obtained by running the command
```
mbeezy freestyle [output_size]
```
with the optional parameter `output_size` being the length of the output.
