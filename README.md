# PPOCoder

Official Implementation of "Execution-based Neural Code Generation using Proximal Policy Optimization"

## Environment Installation
```
pip install -r requirements.txt
```


## Datasets

Ee finetune/evaluate models on the following major dataset benchmarks for different code generation tasks:

* **CodeSearchNet(CSN)**: The dataset is available [here](https://github.com/github/CodeSearchNet##data-details)
* **XLCoST**: The dataset is available [here](https://github.com/reddy-lab-code-research/XLCoST)
* **APPS**: The dataset is available [here](https://github.com/hendrycks/apps)
* **MBPP**: The dataset is available [here](https://github.com/google-research/google-research/tree/master/mbpp)

We preprocess the data and construct input/output sequences in the same manner as outlined in the original benchmark papers for all benchmarks. Unzip and place all benchmarks in the `data` folder.
