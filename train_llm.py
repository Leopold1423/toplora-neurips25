import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['WANDB_MODE'] = 'disabled'
import yaml
import torch
import socket
import transformers
from datetime import datetime
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import get_log
from utils.dev_parse import load_config, print_dict_paths
from utils.finetune_utils import get_data
from utils.evaluate_utils import  MATH_DATASETS, COMMONSENSE_DATASETS, evaluate_dataset
from utils.train_utils import set_global_seed, convert_target_modules, get_trainable_params_numbers, print_delta_time, convert_lora_params_dtype

from mypeft.toplora import replace_toplora_linear


if __name__ == "__main__":
    config_dict, config = load_config("config/lora.yaml")
    logger = get_log(config.output_dir, "setting")
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
    
    ## prepare
    server_name = socket.gethostname()
    logger.info(f"server_name: {server_name}")
    print_dict_paths(config_dict, logger)
    train_config = config.train_config
    test_config = config.test_config
    lora_config = config.lora_config
    lora_config.target_modules = convert_target_modules(lora_config.target_modules)
    set_global_seed(config.seed)
    ft_start_time = datetime.now()
    
    ## model
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # allow batched inference
    
    ## lora
    peft_config = LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        use_dora=getattr(lora_config, "use_dora", False),
    )
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    
    ## toplora
    if getattr(lora_config, "use_toplora", False):
        replace_toplora_linear(model)

    convert_lora_params_dtype(model, dtype=lora_config.dtype)
    logger.info(model)
    rate = get_trainable_params_numbers(model, path=os.path.join(config.output_dir, "num_params.json"))
    logger.info(rate)

    ## print trainable names
    logger.info("### Trainable Parameters:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad is True:
            logger.info(f"{name}: {parameter.dtype}")

    ## finetune
    if config.finetune:
        train_data, val_data = get_data(
            config.data_path, 
            train_config.val_set_size, 
            tokenizer, 
            train_config.max_length, 
            train_config.train_on_inputs
            )
        training_args = transformers.TrainingArguments(
            warmup_steps=100,
            save_only_model=True,
            per_device_train_batch_size=train_config.micro_batch_size,
            gradient_accumulation_steps=train_config.batch_size//train_config.micro_batch_size,
            gradient_checkpointing=train_config.use_gradient_checkpointing,
            num_train_epochs=train_config.num_epochs,
            learning_rate=train_config.lr,
            weight_decay=train_config.weight_decay,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps" if train_config.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=train_config.eval_step if train_config.val_set_size > 0 else None,
            save_steps=train_config.save_step,
            output_dir=os.path.join(config.output_dir, "finetuned_result"),
            save_total_limit=1,
            load_best_model_at_end=True if train_config.val_set_size > 0 else False,
            report_to=None,
            seed=config.seed,
            )
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        trainer.train()
        model.save_pretrained(os.path.join(config.output_dir, "finetuned_result"))
        print_delta_time(ft_start_time, logger)

    ## evaluate
    if config.evaluate:
        model.eval()
        if test_config.merge:  # some methods do not support merge
            model = model.merge_and_unload()
        torch.cuda.empty_cache()
        
        test_datasets = MATH_DATASETS if "math" in config.data_path else COMMONSENSE_DATASETS
        if test_config.test_dataset_ids != "":
            test_datasets = [test_datasets[int(i)] for i in test_config.test_dataset_ids]
        
        set_global_seed(config.seed)
        eval_start_time = datetime.now()
        for test_dataset in test_datasets:
            accuracy = evaluate_dataset(model, tokenizer, test_dataset, test_config.test_batch_size, os.path.join(config.output_dir, "evaluated_result"))
            if accuracy < 0.01:
                logger.info("stop evaluate due to poor accuracy.")
                break
        print_delta_time(eval_start_time, logger)
