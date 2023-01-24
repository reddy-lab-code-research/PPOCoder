# PPOCoder

Official Implementation of "Execution-based Neural Code Generation using Proximal Policy Optimization"
<!-- 
## Overview

 -->

## Environment Installation
To run the code, install the dependencies in requirements.txt.
```
pip install -r requirements.txt
```


## Datasets
We finetune/evaluate models on the following major dataset benchmarks for different code generation tasks:

* **CodeSearchNet(CSN)** is available [here](https://github.com/github/CodeSearchNet##data-details)
* **XLCoST** is available [here](https://github.com/reddy-lab-code-research/XLCoST)
* **APPS** is available [here](https://github.com/hendrycks/apps)
* **MBPP** is available [here](https://github.com/google-research/google-research/tree/master/mbpp)

We preprocess the data and construct input/output sequences in the same manner as outlined in the original benchmark papers. Unzip and place all benchmarks in the `data` folder.


## Run
We have created `run.sh` script to execute PPO-based PL model fine-tuning based on the compiler signal. To run the script for different code generation tasks, configure the following parameters:

<!-- |   **Parameters**  |                                              **Description**                                             |       **Example Values**       |
|:-----------------:|:--------------------------------------------------------------------------------------------------------:|:------------------------------:|
| `l1`        | Source Language                                                                     | java,python,cpp,cs,nl,php,csharp,c |
| `l2`    | Target Language                                  | models/codet5_tokenizer/       |
 -->


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



<!-- ## Citation -->
<!-- If you find the paper or the source code useful to your projects, please cite the following bibtex: 
<pre>
@inproceedings{
	le2022coderl,
	title={Code{RL}: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning},
	author={Hung Le and Yue Wang and Akhilesh Deepak Gotmare and Silvio Savarese and Steven Hoi},
	booktitle={Advances in Neural Information Processing Systems},
	editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
	year={2022},
	url={https://openreview.net/forum?id=WaGvb7OzySA}
}
</pre> -->