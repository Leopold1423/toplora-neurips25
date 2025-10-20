# TopLoRA

Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation [NeurIPS25]

## Environment Setup
Please execute the following command to install the necessary dependencies:
```
pip install -r requirements.txt
```

## Dataset 
Please download the dataset from [LLM_Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). You need to move math_10k.json and commonsense_170k.json to the dataset directory, and then copy the dataset directory to this directory
```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters
cp LLM-Adapters/ft-training_set/math_10k.json LLM-Adapters/dataset
cp LLM-Adapters/ft-training_set/commonsense_170k.json LLM-Adapters/dataset
cp -r LLM-Adapters/dataset ./
rm -r LLM-Adapters
```

## Run the Experiment
The hyperparameter settings are written in the corresponding yaml file in the config directory. You can quickly run each method by specifying the corresponding yaml file, for example
```
CUDA_VISIBLE_DEVICES=0 python train_llm.py -f "config/lora.yaml"
CUDA_VISIBLE_DEVICES=0 python train_llm.py -f "config/toplora.yaml"
```

## Citation

```
@inproceedings{li2025toplora,
  title={Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation},
  author={Li, Shiwei and Luo, Xiandi and Wang, Haozhao and Tang, Xing and Cui, Ziqiang and Liu, Dugang and Li, Yuhua and He, Xiuqiang and Li, Ruixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
