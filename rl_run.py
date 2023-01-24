import os
import torch
import numpy as np
import datetime
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (tree_to_token_index,
                   tree_to_token_nodes,
                   index_to_code_token,
                   tree_to_variable_index, 
                   detokenize_code)
from tree_sitter import Language, Parser
from reward import remove_special_tokens, tree_sitter_full_compile, get_reward
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from model import CodeT5HeadWithValueModel, respond_to_batch
from transformers import RobertaTokenizer
from ppo import PPOTrainer
import torch
from itertools import cycle
from tqdm import tqdm
from bleu import _bleu
from utils import extract_structure, Example, read_examples, convert_examples_to_features, InputFeatures
import argparse


parser = argparse.ArgumentParser()
## Required parameters  
parser.add_argument("--l1", default=None, type=str,
                        help="source language")  
parser.add_argument("--l2", default=None, type=str,
                    help="target language") 
parser.add_argument("--asp", default=2, type=int,
                    help="action space")  
parser.add_argument("--ns", default=5, type=int,
                    help="num syn samples") 
parser.add_argument("--data_path", default=None, type=str,
                    help="data parent directory")  
parser.add_argument("--output_path", default=None, type=str,
                    help="output directory")
parser.add_argument("--load_model_path", default=None, type=str,
                    help="path to load models")
parser.add_argument("--baseline_output_path", default=None, type=str,
                    help="path to load models")
parser.add_argument("--max_source_length", default=400, type=int,
                    help="maximum source length")
parser.add_argument("--max_target_length", default=400, type=int,
                    help="maximum target length")
parser.add_argument("--train_batch_size", default=16, type=int,
                    help="train_batch_size")
parser.add_argument("--test_batch_size", default=48, type=int,
                    help="test_batch_size")
parser.add_argument("--train_epochs", default=1000000, type=int,
                    help="test_batch_size")
parser.add_argument("--run", default=1, type=int,
                    help="run ID")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--kl_coef", type=float, default=0.05, help="KL Coefficient")
parser.add_argument("--kl_target", type=float, default=1, help="Adaptive KL Target")
parser.add_argument("--vf_coef", type=float, default=1e-3, help="Coefficient of the Value Error")
  

args = parser.parse_args()
args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_parent_dir = args.data_path
dir_dict = {'javascript':'Javascript', 'java':'Java', 'c_sharp':'C#', 'php':'PHP', 'python':'Python', 'c':'C', 'cpp':'C++'}
end_dict = {'javascript':'js', 'java':'java', 'c_sharp':'cs', 'php':'php', 'python':'py', 'c':'c', 'cpp':'cpp'}
l1, l2 = args.l1, args.l2
data_dir = data_parent_dir + '/' + dir_dict[l1] + '-' + dir_dict[l2] + '/'
template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
template = template.replace('XXX', dir_dict[l1]).replace('YYY', dir_dict[l2])
if not(os.path.exists(data_dir)):
    data_dir = data_parent_dir + '/' + dir_dict[l2] + '-' + dir_dict[l1] + '/'
    template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
    template = template.replace('XXX', dir_dict[l2]).replace('YYY', dir_dict[l1])
train_filename = template.replace('xxx', end_dict[l1]).replace('yyy', end_dict[l2])
dev_filename = train_filename.replace('train', 'val')
test_filename = train_filename.replace('train', 'test')
baseline_output_dir = args.baseline_output_path + '/'+l1+'-'+l2+'/'
load_model_path = args.load_model_path
output_dir = args.output_path + '/'+l1+'-'+l2+'/'

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
    'c':DFG_csharp,
    'cpp':DFG_csharp,}
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser


def get_dataset(features):
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)  
    indices = torch.arange(len(features))
    data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask,indices)
    return data

#load models
model = CodeT5HeadWithValueModel()
model.load_base_model(args.load_model_path)
model.to(args.device)

model_ref = CodeT5HeadWithValueModel()
model_ref.load_base_model(args.load_model_path)
model_ref.to(args.device)
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", do_lower_case=False)
ppo_config = {"batch_size": args.train_batch_size, 'eos_token_id': tokenizer.eos_token_id, 'lr':args.lr, "adap_kl_ctrl": True, 'init_kl_coef':args.kl_coef,"target":args.kl_target, "vf_coef":args.vf_coef}
ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)

#load features
train_examples = read_examples(args.train_filename, args)
train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
dev_examples = read_examples(args.dev_filename, args)
dev_features = convert_examples_to_features(dev_examples, tokenizer, args, stage='train')
test_examples = read_examples(args.test_filename, args)
test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='train')


def test(epoch,features, dataloader, prefix):
    pbar = dataloader
    pred_ids = []
    pred_ids_ref = []
    indices = []
    nerrors = 0
    nerrors_ref = 0
    with torch.no_grad():
        for batch in pbar:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask,target_ids,target_mask,ind = batch
            preds = respond_to_batch(model, source_ids, source_mask, max_target_length=args.max_target_length, \
                                                     top_k=args.asp, top_p=1.0)[:,1:]
            preds_ref = respond_to_batch(model_ref, source_ids, source_mask, max_target_length=args.max_target_length, \
                                                     top_k=args.asp, top_p=1.0)[:,1:]
            nerrors += sum(get_reward(code_ids=preds, code_ref_ids=preds_ref,gold_ids=target_ids, tokenizer=tokenizer)[4])
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
            nerrors_ref += sum(get_reward(code_ids=preds_ref,code_ref_ids=preds_ref,gold_ids=target_ids, tokenizer=tokenizer)[5])
            top_preds = list(preds_ref.cpu().numpy())
            pred_ids_ref.extend(top_preds)
            indices.extend(list(ind.cpu().numpy()))
   
    p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                              for id in pred_ids]
    p_ref = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                              for id in pred_ids_ref]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(path,prefix+".model"+ "_ep%d"%(epoch) ),'w') as f_model, \
            open(os.path.join(path,prefix+".gold"+ "_ep%d"%(epoch) ),'w') as f_gold, \
              open(os.path.join(path,prefix+".model_ref"+ "_ep%d"%(epoch) ),'w') as f_ref:
                    for pred,ref,i in zip(p,p_ref,indices):
                        f_model.write(pred+'\n')
                        f_ref.write(ref+'\n')   
                        f_gold.write(features[i].target+'\n')
                   
    return nerrors, nerrors_ref


#datasets
train_data = get_dataset(train_features)
train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
dev_data = get_dataset(dev_features)
dev_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=False)
test_data = get_dataset(test_features)
test_dataloader = DataLoader(test_data, batch_size=args.train_batch_size, shuffle=False)


#training
nsteps = 0
total_nerrors = 0
total_rewards = 0
total_nnodes = 0
total_nerrors_ref = 0
total_nnodes_ref = 0
total_seen = 0
n_syn_samples = args.ns
for ep in range(args.train_epochs):
    for samp in range(n_syn_samples):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask,target_ids,target_mask, _ = batch
     
            response_ids  = torch.clone(respond_to_batch(model, source_ids, source_mask, \
                                                        max_target_length=args.max_target_length, \
                                                        top_k=args.asp, top_p=1.0).detach()[:,1:])
            response_codes = tokenizer.batch_decode(response_ids, skip_special_tokens=True, \
                                                    clean_up_tokenization_spaces=False)

            response_ids_ref  = torch.clone(respond_to_batch(model_ref, source_ids, source_mask, \
                                                        max_target_length=args.max_target_length, \
                                                        top_k=args.asp, top_p=1.0).detach()[:,1:])     
  
            reward,mean_rate,mean_ast_match,mean_dfg_match, num_errors,num_errors_ref, num_nodes,num_nodes_ref  = get_reward(lang = args.l2, code_ids=response_ids,code_ref_ids=response_ids_ref, gold_ids=target_ids, tokenizer=tokenizer)
            
            total_rewards += sum([reward.sum(axis=-1).tolist()])
            total_nerrors += sum(num_errors)
            total_nnodes += sum(num_nodes)
            total_nerrors_ref += sum(num_errors_ref)
            total_nnodes_ref += sum(num_nodes_ref)
            total_seen += len(source_ids)

            pbar.set_description('Avg # errors per sample:'+str(round(total_nerrors/total_seen, 5)))

            #PPO Step
            train_stats = ppo_trainer.step(source_ids, source_mask, response_ids,response_ids_ref, reward.to(args.device))
            
            
            mean_kl = train_stats['objective/kl']
            mean_entropy = train_stats['objective/entropy']
            loss, pg_loss, vf_loss = train_stats['ppo/loss/total'], train_stats['ppo/loss/policy'], train_stats['ppo/loss/value']
            mean_advg, mean_return,mean_val = train_stats['ppo/policy/advantages_mean'], train_stats['ppo/returns/mean'], train_stats['ppo/val/mean']

            nsteps += 1

            #save the results
            with open(output_dir+'results/'+l1+'-'+l2+'.csv', 'a') as f:
                f.write( datetime.datetime.now().strftime("%H:%M:%S") +  
                        ',' + str(args.run)+
                        ',' + str(args.train_batch_size)+
                        ',' + str(args.max_source_length)+
                        ',' + str(args.max_target_length)+
                        ',' + str(args.lr)+ 
                        ',' + str(ep)+ 
                        ',' + str(nsteps)+ 
                        ',' + str(round(sum([reward.sum(axis=-1).tolist()])/len(source_ids), 4))+
                        ',' + str(round(sum(num_errors)/len(source_ids), 4))+
                        ',' + str(round(sum(num_errors_ref)/len(source_ids), 4))+
                        ',' + str(round(sum(num_nodes)/len(source_ids), 4))+
                        ',' + str(round(sum(num_nodes_ref)/len(source_ids),4)) +
                        ',' + str(mean_kl) +
                        ',' + str(mean_entropy) + 
                        ',' + str(loss.item()) + 
                        ',' + str(pg_loss.item()) + 
                        ',' + str(vf_loss.item()) + 
                        ',' + str(mean_advg.item()) + 
                        ',' + str(mean_return.item()) + 
                        ',' + str(mean_val.item()) + 
                        ',' + str(mean_rate) +
                        ',' + str(mean_ast_match) +
                        ',' + str(mean_dfg_match)
                        + '\n')


    path = output_dir
    path = os.path.join(path, 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model_ep%d.bin"%(ep))
    torch.save(model_to_save.state_dict(), output_model_file)
    
    train_dataloader2 = DataLoader(train_data, batch_size=args.test_batch_size, shuffle=False)
    test_dataloader2 = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    nerrors, nerrors_ref = test(ep,train_features, train_dataloader2, 'train')
    nerrors_test, nerrors_ref_test = test(ep,test_features, test_dataloader2, 'test')
      
