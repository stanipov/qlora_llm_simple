from transformers import TrainerCallback
import os
from torch import bfloat16
import logging
import datetime as dt
import sys

def set_logger():
    """ Sets up a stdout logger """
    today = dt.datetime.today()
    dt_str = f"{today.month:02d}-{today.day:02d}-{today.year}"

    logFormatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)8s] [%(levelname)-8s] %(message)s"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logFormatter)
    ch.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)
    logger = logging.getLogger(__name__)
    return logger


class PeftSavingCallback(TrainerCallback):
    """ Modify what is saved every save step """
    def on_save(self, args, state, control, **kwargs):

        rm_opt_state = kwargs.pop('rm_opt_state', True)
        rm_model = kwargs.pop('rm_orig_model', True)

        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        # if original model is saver. remove
        if rm_model:
            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

        # remove optimizer state dict
        if rm_opt_state:
            if "optimizer.pt" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "optimizer.pt"))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return trainable_params, all_param

def gen_train_cfg():
    cfg = {}
    # location to download the base model
    cfg['model_cache'] = '/models/model_cache'
    # random seed
    cfg['random_seed'] = 19880122

    # dataset related parameters
    # location of the dataset
    cfg['src'] = '/home/sf/data/py_proj/2023/nlp_playground/datasets/chatml_eds-5-4-2/merged-ds'
    #'/home/sf/data/py_proj/2023/nlp_playground/datasets/chatml_eds-5-4-2/merged-ds'

    # fraction of the whole dataset to be used for validation
    cfg['val_frac'] = 2e-2

    # tokens
    cfg['special_tokens'] = {
        'bos_token': '<|im_end|>'
    }
    cfg['extra_tokens'] = ['<|im_start|>']
    cfg['padding'] = True
    cfg['chat_template'] = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    cfg['experiment_name'] = 'e8-1_ds5-4-2'

    cfg['model_name'] = "mistralai/Mistral-7B-v0.1"

    # train
    cfg['num_train_epochs'] = 2
    cfg['per_device_train_batch_size'] = 4
    cfg['per_device_eval_batch_size'] = 4
    cfg['per_device_eval_batch_size'] = 4
    cfg['gradient_accumulation_steps'] = 8
    cfg['gradient_checkpointing'] = False
    cfg['output_dir'] = f"/models/finetune/llm/experiments/{cfg['experiment_name']}"

    cfg['save_steps'] = 250
    cfg['logging_steps'] = 20
    cfg['eval_steps'] = 5000
    cfg['save_total_limit'] = None
    cfg['max_grad_norm'] = 0.45
    cfg['learning_rate'] = 1e-5
    cfg['weight_decay'] = 0.01
    cfg['optim'] = "paged_adamw_32bit"
    cfg['lr_scheduler_type'] = 'cosine'
    cfg['max_steps'] = -1
    cfg['warmup_ratio'] = 0.03
    cfg['group_by_length'] = True
    cfg['max_seq_length'] = 1500
    cfg['packing'] = False
    cfg['device_map'] = {"": 0}

    # LoRA
    cfg['lora_r'] = 256
    cfg['lora_alpha'] = cfg['lora_r'] * 2
    cfg['lora_dropout'] = 0.05
    cfg['use_4bit'] = True
    cfg['bnb_4bit_compute_dtype'] = bfloat16
    cfg['bnb_4bit_quant_type'] = "nf4"
    cfg['use_nested_quant'] = True
    cfg['fp16'] = False
    cfg['bf16'] = True
    cfg['tf32'] = True
    cfg['target_modules'] = [

        # tokens
        "embed_tokens",

        # Decoder
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",

        # MLP
        #"gate_proj",
        #"up_proj"
        #"down_proj",

        # Head
        # "lm_head"
    ]


    # wandb
    cfg['wandb'] = {
        'enabled': True,
        'msg': f"""Conversational tune of the base model (Mistral 7B). Do not tune MLP layers.""",
        'proj_name_core' : 'mistral-rude-bot',

    }
    cfg['wandb']['proj_name'] = f"{cfg['wandb']['proj_name_core']}-{cfg['experiment_name']}"

    return cfg
