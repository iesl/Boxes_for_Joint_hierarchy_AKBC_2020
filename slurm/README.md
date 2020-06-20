# Instructions for training on gypsum
Currently gypsum does not have cuda101 hence the default latest version of compiled torch (ie detaul torch 1.3.0) will not use the gpu. Use the following command to install the correct binary of torch:

```
pip3 install torch==1.3.0+cu92 torchvision==0.4.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
Check if torch is detecting the gpu by acquireing an interactive session, activating the virtual env and calling `torch.cuda.is_available()`.