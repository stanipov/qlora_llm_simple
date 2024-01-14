import logging
import datetime as dt
import sys

import os
from functools import partial
import torch
import torch.nn as nn

from transformers import TrainerCallback
import transformers
import torch

import pandas as pd
import os
from datasets import load_dataset, Dataset

from peft import PeftModel
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import wandb

def conversation2prompt(conversation, bos_token, eos_token):
    prompt = ''
    for line in conversation:
        prompt += f'{bos_token}{line["role"]}\n{line["content"]}{eos_token}'
    return prompt



from utils import gen_train_cfg, print_trainable_parameters, PeftSavingCallback, set_logger

if __name__ == "__main__":
    # slow drive:
    # model_cache = '/ext4/model_cache/hf_cache'
    # fast drive:
    #model_cache = '/models/model_cache'
    logger = set_logger()
    cfg = gen_train_cfg()
    RANDOM_SEED = cfg['random_seed']
    model_cache = cfg['model_cache']
    # dataset parameters
    src = cfg['src']
    val_frac = cfg['val_frac']

    #tokens
    special_tokens = cfg.pop('special_tokens', {})
    extra_tokens = cfg.pop('extra_tokens', [])
    _padding = cfg['padding']
    chat_template = cfg['chat_template']

    experiment_name = cfg['experiment_name']
    model_name = cfg['model_name']

    # train
    num_train_epochs = cfg['num_train_epochs']
    per_device_train_batch_size = cfg['per_device_train_batch_size']
    per_device_eval_batch_size = cfg['per_device_eval_batch_size']
    gradient_accumulation_steps = cfg['gradient_accumulation_steps']
    gradient_checkpointing = cfg['gradient_checkpointing']
    output_dir = cfg['output_dir']
    save_steps = cfg['save_steps']
    logging_steps = cfg['logging_steps']
    eval_steps = cfg['eval_steps']
    save_total_limit = cfg['save_total_limit']
    max_grad_norm = cfg['max_grad_norm']
    learning_rate = cfg['learning_rate']
    weight_decay = cfg['weight_decay']
    optim = cfg['optim']
    lr_scheduler_type = cfg['lr_scheduler_type']
    max_steps = cfg['max_steps']
    warmup_ratio = cfg['warmup_ratio']
    group_by_length = cfg['group_by_length']
    max_seq_length = cfg['max_seq_length']
    packing = cfg['packing']
    device_map = cfg['device_map']

    # LoRA
    lora_r = cfg['lora_r']
    lora_alpha = cfg['lora_alpha']
    lora_dropout = cfg['lora_dropout']
    use_4bit = cfg['use_4bit']
    bnb_4bit_compute_dtype = cfg['bnb_4bit_compute_dtype']
    bnb_4bit_quant_type = cfg['bnb_4bit_quant_type']
    use_nested_quant = cfg['use_nested_quant']
    fp16 = cfg['fp16']
    bf16 = cfg['bf16']
    tf32 = cfg['tf32']

    # set env variables to ensure model is downloaded into the specified folder
    os.environ['HF_DATASETS_CACHE'] = model_cache
    os.environ['TRANSFORMERS_CACHE'] = model_cache
    os.environ['HUGGINGFACE_HUB_CACHE '] = model_cache
    os.environ['TRANSFORMERS_CACHE'] = model_cache
    os.environ['HF_HOME'] = model_cache
    os.environ['XDG_CACHE_HOME'] = model_cache

    # Message to WanDB
    wandbmsg = cfg['wandb'].pop('msg', '')

    # BnB config
    logger.info('Setting BnB config.')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,

    )

    logger.info('Setting LoRA config.')
    # LoRA config
    target_modules = cfg['target_modules']
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        tf32=tf32,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb" if cfg['wandb']['enabled'] else None,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=save_total_limit
    )

    # set the wandb project where this run will be logged
    if cfg['wandb']['enabled']:
        wandb.login()

        proj_name_core = cfg['wandb']['proj_name_core']
        proj_name = cfg['wandb']['proj_name']
        os.environ["WANDB_PROJECT"] = proj_name

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"] = "true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"] = "false"

        run = wandb.init(
            # Set the project where this run will be logged
            project=proj_name,
            # Track hyperparameters and run metadata
            config={
                "quantization": bnb_config,
                "LORA": lora_config,
                'training': training_arguments,
                'notes': wandbmsg
            })

    # setting up the tokenizer
    logger.info(f'Setting up the tokenizer.')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,
                                              cache_dir=model_cache, max_length=max_seq_length)
    if special_tokens:
        t = tokenizer.add_special_tokens(special_tokens)
        logger.info(f'Added {t} special tokens.')
    if extra_tokens:
        t = tokenizer.add_tokens(extra_tokens)
        logger.info(f'Added {t} extra tokens.' )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length
    tokenizer.padding_side = 'right'
    if chat_template:
        tokenizer.chat_template = chat_template

    # loading the model
    logger.info(f'Loading the model.')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_cache,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()

    # prepare for PEFT
    logger.info('Preparing for PEFT training')
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    trainable_params, all_param = print_trainable_parameters(model)
    t = f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}"
    logger.info(t)

    # Setting up the training and validation datasets
    # Loading
    logger.info('Preparing dataset')
    DS = Dataset.load_from_disk(src)
    split = DS.train_test_split(test_size=val_frac, seed=RANDOM_SEED)
    ds1 = split['train']
    ds2 = split['test']

    # Applying chat template
    logger.info('Applying chat template to the train data')
    train_data = ds1.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'])}, batched=False)
    logger.info('Applying chat template to the validation data')
    val_data = ds2.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'])}, batched=False)

    # drop useless column
    train_data = train_data.remove_columns(['messages'], )
    val_data = val_data.remove_columns(['messages'], )

    # show info
    logger.info(f'Train len: {len(train_data)}')
    logger.info(f'Eval len: {len(val_data)}')

    # Setting up the trainer
    logger.info('Setting the SFTTrainer')
    callbacks = [PeftSavingCallback()]
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=packing,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        dataset_text_field='text',
        args=training_arguments,
        callbacks=callbacks
    )

    # New model path
    new_model_dst = os.path.join(output_dir, 'final')
    os.makedirs(new_model_dst, exist_ok=True)
    logger.info(f'Final adapter will be save in: {new_model_dst}')

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    logger.info('Saving the final adapter')
    trainer.model.save_pretrained(new_model_dst)
    logger.info(f'Saving the tokenizer in {new_model_dst}')
    tokenizer.save_pretrained(new_model_dst)
    try:
        trainer.tokenizer.save_pretrained(new_model_dst)
    except Exception as e:
        logger.warning(e)