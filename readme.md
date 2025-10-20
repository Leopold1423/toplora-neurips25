# TopLoRA

Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation [NeurIPS 2025]

## Environment Setup
Please execute the following command to install the necessary dependencies:
```
pip install -r requirements.txt
```

## Dataset 
Please download the datasets from [LLM_Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), place them in the dataset directory, and then copy this directory to the current project.
```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters
cp LLM-Adapters/ft-training_set/math_10k.json LLM-Adapters/dataset
cp LLM-Adapters/ft-training_set/commonsense_170k.json LLM-Adapters/dataset
cp -r LLM-Adapters/dataset ./
rm -rf LLM-Adapters
```

## Run the Experiment
To run a method, specify its corresponding yaml file from the config directory. For example:
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
