#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train_llm.py -f "config/lora.yaml"
CUDA_VISIBLE_DEVICES=2 python train_llm.py -f "config/toplora.yaml"