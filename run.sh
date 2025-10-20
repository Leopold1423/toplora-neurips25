#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python train_llm.py -f "config/lora.yaml"
CUDA_VISIBLE_DEVICES=4 python train_llm.py -f "config/toplora.yaml"