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
from reward import remove_special_tokens, tree_sitter_full_compile, get_binary_compilation_reward, get_reward
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from codet5 import CodeT5HeadWithValueModel, respond_to_batch
from transformers import RobertaTokenizer
from ppo import PPOTrainer
import torch
from itertools import cycle
from tqdm import tqdm
from bleu import _bleu
from core import extract_structure
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
  

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]="1"
data_parent_dir = args.data_path
dir_dict = {'javascript':'Javascript', 'java':'Java', 'c_sharp':'C#', 'php':'PHP', 'python':'Python', 'c':'C', 'cpp':'C++'}
end_dict = {'javascript':'js', 'java':'java', 'c_sharp':'cs', 'php':'php', 'python':'py', 'c':'c', 'cpp':'cpp'}
l1, l2 = args.l1, args.l2
data_dir = data_parent_dir + dir_dict[l1] + '-' + dir_dict[l2] + '/'
template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
template = template.replace('XXX', dir_dict[l1]).replace('YYY', dir_dict[l2])
if not(os.path.exists(data_dir)):
    data_dir = data_parent_dir + dir_dict[l2] + '-' + dir_dict[l1] + '/'
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
# np.random.seed(42)
# torch.manual_seed(42)

####################################
class Args():
    def __init__(self):
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.baseline_output_dir = baseline_output_dir
        self.load_model_path = load_model_path
        self.max_source_length = 400#400
        self.max_target_length = 400#400
        self.train_batch_size = 16#16
        self.test_batch_size = 48#16
        self.train_epochs = 1000000
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        self.output_dir = output_dir
        self.reward_type = 'n_nodes-10n_errors'
        self.reward_id = 2
        self.run = 4
        self.loss_W = 10
        self.lr = 1e-5
        self.kl_coef = 1
        self.reward_W = 0.01
        self.action_space = args2.asp
        self.source_lang = l1
        self.target_lang = l2
        self.max_ast_length = 250
        self.max_ast_depth = 12
        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)  
args = Args()
####################################
#num errors orig code (base codet5)
# def get_num_errors(filepath):
#     codes = open(filepath).readlines()
#     codes = [remove_special_tokens(code[:-1]) for code in codes]
#     num_errors = [tree_sitter_full_compile(code) for code in codes]
#     return num_errors
# for filename in ['dev.output', 'test_0.output']:
#     num_errors = get_num_errors(args.baseline_output_dir+filename)
#     print (filename, num_errors)
#     print("Mean Nb. Errors in %s:"%(filename), np.array(num_errors)[:,0].mean())
#     print("Sum Nb. Errors in %s:"%(filename), np.array(num_errors)[:,0].sum())
####################################
class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 source_orig,
                 target_orig
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.source_orig = source_orig
        self.target_orig = target_orig

def read_examples(filename, args):
    examples=[]
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                line1=line1.strip().replace('▁', '_')
                line2=line2.strip().replace('▁', '_')
                if (args.source_lang=='php') and not(line1.startswith('<?php')):
                    line1 = '<?php '+line1
                if (args.target_lang=='php') and not(line2.startswith('<?php')):
                    line2 = '<?php '+line2
                    
                orig_line1, orig_line2 = line1, line2
                
                if args.source_lang=='python':
                    line1 = detokenize_code(line1)
                else:
                    line1 = line1.replace('NEW_LINE', '\n')
                if args.target_lang=='python':
                    line2 = detokenize_code(line2)
                else:
                    line2 = line2.replace('NEW_LINE', '\n')

                examples.append(
                Example(idx = idx,
                        source=line1,
                        target=line2,
                        source_orig = orig_line1,
                        target_orig = orig_line2) )
                idx+=1
    return examples

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 target):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask   
        self.target = target
  
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source_orig)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target_orig)[:args.max_target_length-1]
        target_tokens = target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        # target_ids+=[-100]*padding_length
        #MODIFIED
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        features.append(InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 example.target_orig))
    # breakpoint()
    return features

def get_dataset(features):
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)  
    indices = torch.arange(len(features))
    data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask,indices)
    return data

#########################################
model = CodeT5HeadWithValueModel()
model.load_base_model(args.load_model_path)
# model = torch.nn.DataParallel(model,device_ids = [0,1,2,3])
model.to(args.device)

model_ref = CodeT5HeadWithValueModel()
model_ref.load_base_model(args.load_model_path)
# model_ref = torch.nn.DataParallel(model_ref,device_ids = [0,1,2,3])
model_ref.to(args.device)
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", do_lower_case=False)
ppo_config = {"batch_size": args.train_batch_size, 'eos_token_id': tokenizer.eos_token_id, 'lr':args.lr, "adap_kl_ctrl": True, 'init_kl_coef':args.kl_coef,"target":0.01, "vf_coef":1e-3,"node_loss_coef":1e-12, 'reward_coef':1}
ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)

train_examples = read_examples(args.train_filename, args)
train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
dev_examples = read_examples(args.dev_filename, args)
dev_features = convert_examples_to_features(dev_examples, tokenizer, args, stage='train')
test_examples = read_examples(args.test_filename, args)
test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='train')
# train_toklens_source = []
# train_toklens_target = []
# test_toklens_source = []
# test_toklens_target = []
# for i in range(len(train_examples)):
#     train_toklens_source.append(len(train_examples[i].source_orig))
#     train_toklens_target.append(len(train_examples[i].target_orig))
# for j in range(len(test_examples)):
#     test_toklens_source.append(len(test_examples[j].source_orig))
#     test_toklens_target.append(len(test_examples[j].target_orig))
# breakpoint()

######################################################################
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
            # top_k = tokenizer.vocab_size #larger action space 
            #top_k = args.action_space
            preds = respond_to_batch(model, source_ids, source_mask, max_target_length=args.max_target_length, \
                                                     top_k=args.action_space, top_p=1.0)[:,1:]
            preds_ref = respond_to_batch(model_ref, source_ids, source_mask, max_target_length=args.max_target_length, \
                                                     top_k=args.action_space, top_p=1.0)[:,1:]
            nerrors += sum(get_reward(code_ids=preds, code_ref_ids=preds_ref,gold_ids=target_ids, tokenizer=tokenizer)[1])
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
            nerrors_ref += sum(get_reward(code_ids=preds_ref,code_ref_ids=preds_ref,gold_ids=target_ids, tokenizer=tokenizer)[1])
            top_preds = list(preds_ref.cpu().numpy())
            pred_ids_ref.extend(top_preds)
            indices.extend(list(ind.cpu().numpy()))
            
#             print (get_reward(code_ids=preds, tokenizer=tokenizer)[1])
#             codes = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) \
#                      for id in top_preds]
#             print (codes)
#             print ([tree_sitter_full_compile(remove_special_tokens(code))[0] for code in codes])
  
    p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                              for id in pred_ids]
    p_ref = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                              for id in pred_ids_ref]

    path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d'%(args.run)+'_as%d'%(args.action_space)+'_ns%d'%(args2.ns)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path,prefix+".model"+ "_ep%d"%(epoch) ),'w') as f_model, \
            open(os.path.join(path,prefix+".gold"+ "_ep%d"%(epoch) ),'w') as f_gold, \
              open(os.path.join(path,prefix+".model_ref"+ "_ep%d"%(epoch) ),'w') as f_ref:
                    for pred,ref,i in zip(p,p_ref,indices):
                        f_model.write(pred+'\n')
                        f_ref.write(ref+'\n')   
                        f_gold.write(features[i].target+'\n')
                        
    # bleu=_bleu(os.path.join(args.output_dir,prefix+".gold"),  os.path.join(args.output_dir,prefix+".model"))
    # bleu_ref=_bleu(os.path.join(args.output_dir,prefix+".gold"),  os.path.join(args.output_dir,prefix+".model_ref"))
    
    # return bleu, bleu_ref, nerrors, nerrors_ref
    return nerrors, nerrors_ref


#####################################################################
# Prepare training data loader  
train_data = get_dataset(train_features)
train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
dev_data = get_dataset(dev_features)
dev_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=False)
test_data = get_dataset(test_features)
test_dataloader = DataLoader(test_data, batch_size=args.train_batch_size, shuffle=False)


# Run training
nsteps = 0
total_nerrors = 0
total_rewards = 0
total_nnodes = 0
total_nerrors_ref = 0
total_nnodes_ref = 0
total_seen = 0
n_syn_samples = args2.ns
for ep in range(args.train_epochs):
    # epoch_nerrors = 0
    # epoch_rewards = 0
    # epoch_nnodes = 0
    # epoch_nerrors_ref = 0
    # epoch_nnodes_ref = 0
    # epoch_seen = 0
    for samp in range(n_syn_samples):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask,target_ids,target_mask, _ = batch
            
            # for samp in range(n_syn_samples):
            # get model response
            response_ids  = torch.clone(respond_to_batch(model, source_ids, source_mask, \
                                                        max_target_length=args.max_target_length, \
                                                        top_k=args.action_space, top_p=1.0).detach()[:,1:])
            response_codes = tokenizer.batch_decode(response_ids, skip_special_tokens=True, \
                                                    clean_up_tokenization_spaces=False)

            response_ids_ref  = torch.clone(respond_to_batch(model_ref, source_ids, source_mask, \
                                                        max_target_length=args.max_target_length, \
                                                        top_k=args.action_space, top_p=1.0).detach()[:,1:])     
            # reward based on nerrors
            # reward2, num_errors,num_errors_ref, num_nodes,num_nodes_ref = get_reward(code_ids=response_ids,code_ref_ids=response_ids_ref, gold_ids=target_ids ,tokenizer=tokenizer)
            # reward based on compilation (binary)
            reward1,mean_comp_rate,mean_ast_match,mean_dfg_match, num_errors,num_errors_ref, num_nodes,num_nodes_ref  = get_binary_compilation_reward(lang = args.target_lang, code_ids=response_ids,code_ref_ids=response_ids_ref, gold_ids=target_ids, tokenizer=tokenizer)
            # reward_ref, num_errors_ref, num_nodes_ref = get_reward(code_ids=response_ids_ref,code_ref_ids=response_ids, tokenizer=tokenizer)
            # epoch_rewards += sum([sum(reward1.tolist()[i]) for i in range(reward1.shape[0])] )
            total_rewards += sum([reward1[reward1>0].sum(axis=-1).tolist()])
            total_nerrors += sum(num_errors)
            total_nnodes += sum(num_nodes)
            total_nerrors_ref += sum(num_errors_ref)
            total_nnodes_ref += sum(num_nodes_ref)
            total_seen += len(source_ids)

            pbar.set_description('Avg # errors per sample:'+str(round(total_nerrors/total_seen, 5)))

            # train model with ppo
            train_stats = ppo_trainer.step(source_ids, source_mask, response_ids,response_ids_ref, reward1.to(args.device))
            # print(train_stats)
            mean_kl = train_stats['objective/kl']
            mean_entropy = train_stats['objective/entropy']
            # kl = train_stats['ppo/policy/policykl']
            loss, pg_loss, vf_loss = train_stats['ppo/loss/total'], train_stats['ppo/loss/policy'], train_stats['ppo/loss/value']
            mean_advg, mean_return,mean_val = train_stats['ppo/policy/advantages_mean'], train_stats['ppo/returns/mean'], train_stats['ppo/val/mean']

            nsteps += 1
            #save all the results
            # with open(direct_path + 'results/baselines.csv', 'a') as f:
            # with open(direct_path + 'results/codet5_ppo_steps.csv', 'a') as f:
            # with open(direct_path + 'results/codet5_ppo_steps5.csv', 'a') as f:
            # with open(direct_path + 'results/codet5_ppo_cpp_python5.csv', 'a') as f:
            with open('results/final/codet5_ppo_'+l1+'-'+l2+'.csv', 'a') as f:
                f.write( datetime.datetime.now().strftime("%H:%M:%S") +  
                        ',CodeT5_PPO' +
                        ',' + 'reward_'+ str(args.reward_id) + 
                        ',' + str(args.run)+
                        ',' + str(args.train_batch_size)+
                        ',' + str(args.max_source_length)+
                        ',' + str(args.max_target_length)+
                        ',' + str(args.lr)+ 
                        ',' + str(ep)+ 
                        ',' + str(nsteps)+ 
                        # ',' + str(total_rewards)+
                        ',' + str(round(sum([reward1[reward1>0].sum(axis=-1).tolist()])/len(source_ids), 4))+
                        # ',' + str(epoch_nerrors) +
                        ',' + str(round(sum(num_errors)/len(source_ids), 4))+
                        # ',' + str(epoch_nerrors_ref) +
                        ',' + str(round(sum(num_errors_ref)/len(source_ids), 4))+
                        # ',' + str(epoch_nnodes) +
                        ',' + str(round(sum(num_nodes)/len(source_ids), 4))+
                        # ',' + str(epoch_nnodes_ref) +
                        ',' + str(round(sum(num_nodes_ref)/len(source_ids),4)) +
                        ',' + str(mean_kl) +
                        ',' + str(mean_entropy) + 
                        ',' + str(loss.item()) + 
                        ',' + str(pg_loss.item()) + 
                        ',' + str(vf_loss.item()) + 
                        ',' + str(mean_advg.item()) + 
                        ',' + str(mean_return.item()) + 
                        ',' + str(mean_val.item()) + 
                        ',' + str(mean_comp_rate) +
                        ',' + str(mean_ast_match) +
                        ',' + str(mean_dfg_match)
                        + '\n')


    #save model after each epoch
    path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d'%(args.run)+'_as%d'%(args.action_space)+'_ns%d'%(args2.ns)+'_v2'
    output_dir = os.path.join(path, 'checkpoints')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model_ep%d.bin"%(ep))
    torch.save(model_to_save.state_dict(), output_model_file)
    
    # compute bleu
    train_dataloader2 = DataLoader(train_data, batch_size=args.test_batch_size, shuffle=False)
    test_dataloader2 = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    nerrors, nerrors_ref = test(ep,train_features, train_dataloader2, 'train')
    nerrors_test, nerrors_ref_test = test(ep,test_features, test_dataloader2, 'test')
      
    # print ('epoch', ep, '# errors', epoch_nerrors, 'BLEU', str(round(bleu,2))+'/'+str(round(bleu_ref,2)), \
    #       'nerrors', str(nerrors)+'/'+str(nerrors_ref))

    print ('epoch', ep,'nerrors', str(nerrors)+'/'+str(nerrors_ref))
