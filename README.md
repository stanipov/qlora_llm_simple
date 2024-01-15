# Supervised fine-tuning of an open-source LLM using QLoRA
This is an example how to fine-tune a LLM using QLoRA and HuggingFace SFTTrainer on a instruct-like dataset.
Intention of this tiny repo is to share how I do fine-tuning of LLM without using extra 
frameworks like Axolotl and on a desktop computer. See [Dataset](#Dataset).

As of this iteration, I did not want to spend time on making a proper config parsing. Change necessary parameters in the gen_train_cfg() function 
from utils.py. Maybe later I add a proper config parsing from a JSON.

The script also add custom chat template to the tokenizer, if specified. Often, base models are not trained for
instructions or conversation. In this case we need add some tokens (depending on what format you are following).

## Usage
Simply run train.py inside your environment. 

## Package requirements
Your Python environment needs:
- Python >=3.9 
- PyTorch >= 2.0
- HuggingFace Transformers >=4.34 (I use tokenizer.apply_chat_template() and I do not remember in which version this has appeared)
- bitsandbytes >= 0.42
- PEFT >=0.7
- TRL >= 0.7.9
- wandb >= 0.16
You can comment out wandb parts if you do not have/want use it. Also, you will probably need to CLI login to wandb before you run this script.

## Hardware
I have a RTX3090 (24Gb VRAM). For a 7B model this is enough to fine-tune with a batch size of 4 and 4 bit quantization and maximal
sequence length of 1500 tokens.

## Dataset
The script assumes input dataset is HuggingFace [Datasets](https://huggingface.co/docs/datasets/index). As of now
the default field is expected to be named "messages". 
The textual data is in [ChatML](https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md)
format. 

## Traning parameters
This section will be gradually populated.
