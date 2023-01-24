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


## Run
The run module supports various parameters input:

```bash
cd PPOCoder
python rl_run.py --run 1 \ #int: run ID 
        --l1 java \ #str: source language
        --l2 cpp \ #str: target language
        --asp 5 \ #int: action space size
        --ns 10 \ #int: number of synthetic samples
        --data_path DATA-PATH \ #str: directory of the dataset
        --outpu_path OUTPUT-PATH \ #str: directory of the output
        --load_model_path LOAD-MODEL-PATH\ #str: path of the base model (before RL)
        --baseline_out_dir BASELINE-PATH \ #str: path of the baseline experiments
        --max_source_length 400 \ #int: maximum length in the source language
        --max_target_length 400 \ #int: maximum length in the target language
        --train_batch_size 32 \ #int: batch size in the training
        --test_batch_size 48 \ #int: batch size in the testing
        --lr 1e-6 \ #float: starting learning rate (before scheduler)
        --kl_coef 0.1 \ #float: initial coefficient of the KL divergence penalty in the reward
        --kl_target 1 \ #float: target of the KL which adaptively controls the KL coefficient 
        --vf_coef 1e-3 #float: coefficient of the vf error in the ppo loss 
```

You can apply this code on different tasks by modifying differnet parameters. 
