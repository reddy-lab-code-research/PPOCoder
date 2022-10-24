import os
import torch
import numpy as np
import datetime
#import wandb


os.environ["CUDA_VISIBLE_DEVICES"]="0"
parent_dir = '/home/grads/parshinshojaee/trl_code/trl_code/datasets/'
dir_dict = {'javascript':'Javascript', 'java':'Java', 'c_sharp':'C#', 'php':'PHP', 'python':'Python', 'c':'C', 'cpp':'C++'}
end_dict = {'javascript':'js', 'java':'java', 'c_sharp':'cs', 'php':'php', 'python':'py', 'c':'c', 'cpp':'cpp'}
l1, l2 = 'php', 'python'
data_dir = parent_dir + dir_dict[l1] + '-' + dir_dict[l2] + '/'
template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
template = template.replace('XXX', dir_dict[l1]).replace('YYY', dir_dict[l2])
if not(os.path.exists(data_dir)):
    data_dir = parent_dir + dir_dict[l2] + '-' + dir_dict[l1] + '/'
    template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
    template = template.replace('XXX', dir_dict[l2]).replace('YYY', dir_dict[l1])
train_filename = template.replace('xxx', end_dict[l1]).replace('yyy', end_dict[l2])
dev_filename = train_filename.replace('train', 'val')
test_filename = train_filename.replace('train', 'test')
baseline_output_dir = 'baselines/codet5/saved_models/'+l1+'-'+l2+'/'
load_model_path = 'baselines/codet5/saved_models/'+l1+'-'+l2+'/checkpoint-best-bleu/pytorch_model.bin'
output_dir = 'saved_models/codet5/saved_models/'+l1+'-'+l2
np.random.seed(42)
torch.manual_seed(42)



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
        self.train_epochs = 1000000
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        self.output_dir = output_dir
        self.reward_type = 'n_nodes-10n_errors'
        self.reward_id = 2
        self.run = 49
        self.loss_W = 10
        self.lr = 1e-6
        self.kl_coef = 100
        self.reward_W = 0.01
        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
        
args = Args()
# print ('# gpu:', args.n_gpu)





from reward import remove_special_tokens, tree_sitter_full_compile, get_binary_compilation_reward

def get_num_errors(filepath):
    codes = open(filepath).readlines()
    codes = [remove_special_tokens(code[:-1]) for code in codes]
    num_errors = [tree_sitter_full_compile(code) for code in codes]
    return num_errors

for filename in ['dev.output', 'test_0.output']:
    num_errors = get_num_errors(args.baseline_output_dir+filename)
    print (filename, num_errors)
    print("Mean Nb. Errors in %s:"%(filename), np.array(num_errors)[:,1].mean())
    print("Sum Nb. Errors in %s:"%(filename), np.array(num_errors)[:,1].sum())


######################################################################

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset

# extract features
class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        
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
            examples.append(
            Example(idx = idx, source=line1, target=line2))
            idx+=1
    return examples

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 target
    ):
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
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
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
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-1]
        target_tokens = target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[-100]*padding_length
        target_mask+=[0]*padding_length   

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 example.target
            )
        )
    return features

def get_dataset(features):
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)  
    indices = torch.arange(len(features))
    data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask,indices)
    return data

######################################################################
from trl.codet5 import CodeT5HeadWithValueModel, respond_to_batch
from transformers import RobertaTokenizer
from trl.ppo import PPOTrainer
import torch
from reward import get_reward

# get models
model = CodeT5HeadWithValueModel()
model.load_base_model(args.load_model_path)
model.to(args.device)

model_ref = CodeT5HeadWithValueModel()
model_ref.load_base_model(args.load_model_path)
model_ref.to(args.device)

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", do_lower_case=False)

# initialize trainer
ppo_config = {"batch_size": args.train_batch_size, 'eos_token_id': tokenizer.eos_token_id, 'lr':args.lr, "adap_kl_ctrl": True, 'init_kl_coef':args.kl_coef,"target":0.1, "vf_coef":1e-9,"node_loss_coef":1e-11 }
ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)

# prepare base model inputs and outputs
train_examples = read_examples(args.train_filename, args)
train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
dev_examples = read_examples(args.dev_filename, args)
dev_features = convert_examples_to_features(dev_examples, tokenizer, args, stage='train')
test_examples = read_examples(args.test_filename, args)
test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='train')


######################################################################
from itertools import cycle
from tqdm import tqdm
from bleu import _bleu
from reward import remove_special_tokens, tree_sitter_full_compile
import numpy as np

def get_reward(code_ids=None, code_ref_ids=None, tokenizer=None):
    code_ids = np.array(code_ids.cpu())
    eos_positions = []
    max_len = code_ids.shape[1]
    for id in code_ids:
        if tokenizer.eos_token_id in id:
            eos_positions.append((id==tokenizer.eos_token_id).argmax())
        else:
            eos_positions.append(max_len)

    codes = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(code_ids, eos_positions)]
    
    codes_ref = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(code_ref_ids, eos_positions)]
        
    codes = [remove_special_tokens(code) for code in codes]
    codes_ref = [remove_special_tokens(code) for code in codes_ref]
    error_node_counts = [tree_sitter_full_compile(code) for code in codes]
    error_node_counts_ref = [tree_sitter_full_compile(code) for code in codes_ref]
    num_errors = [i[0] for i in error_node_counts]
    num_errors_ref = [i[0] for i in error_node_counts_ref]  
    num_nodes = [i[1] for i in error_node_counts]
    num_nodes_ref = [i[1] for i in error_node_counts_ref]
    
    
    # rewards = np.zeros_like(code_ids, dtype=np.float)
    # for i in range(len(rewards)):
    #     # rewards[i, min(eos_positions[i],max_len-1)] = (num_nodes[i]-num_errors[i])*0.001 
    #     # rewards[i, min(eos_positions[i],max_len-1)] = (num_nodes[i]-num_errors[i])
    #     rewards[i, min(eos_positions[i],max_len-1)] = (num_nodes[i]-args.loss_W*num_errors[i])*args.reward_W
        # rewards[i, min(eos_positions[i],max_len-1)] = -num_errors[i]
    
    ###############################
    #ADDED PARSHIN: TO JUST SEE THE PERFORMANCE OF KL TERM   
    rewards = np.zeros_like(code_ids, dtype=np.float)
    # breakpoint()
    # for i in range(len(rewards)):
    #     rewards[i, min(eos_positions[i],max_len-1)] = -args.reward_W*(num_nodes[i] - num_nodes_ref[i])**2
    ###############################
    return torch.Tensor(rewards), num_errors,num_errors_ref, num_nodes, num_nodes_ref


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
                                                     top_k=tokenizer.vocab_size, top_p=1.0)[:,1:]
            preds_ref = respond_to_batch(model_ref, source_ids, source_mask, max_target_length=args.max_target_length, \
                                                     top_k=tokenizer.vocab_size, top_p=1.0)[:,1:]
            nerrors += sum(get_reward(code_ids=preds, code_ref_ids=preds_ref, tokenizer=tokenizer)[1])
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
            nerrors_ref += sum(get_reward(code_ids=preds_ref,code_ref_ids=preds_ref, tokenizer=tokenizer)[1])
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

    path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d'%(args.run)
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


# Run training
nsteps = 0
for ep in range(args.train_epochs):
    epoch_nerrors = 0
    epoch_rewards = 0
    epoch_nnodes = 0
    epoch_nerrors_ref = 0
    epoch_nnodes_ref = 0
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    epoch_seen = 0
    for batch in pbar:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids,source_mask,target_ids,target_mask, _ = batch

        # get model response
        response_ids  = torch.clone(respond_to_batch(model, source_ids, source_mask, \
                                                     max_target_length=args.max_target_length, \
                                                     top_k=tokenizer.vocab_size, top_p=1.0).detach()[:,1:])
        response_codes = tokenizer.batch_decode(response_ids, skip_special_tokens=True, \
                                                clean_up_tokenization_spaces=False)

        response_ids_ref  = torch.clone(respond_to_batch(model_ref, source_ids, source_mask, \
                                                     max_target_length=args.max_target_length, \
                                                     top_k=tokenizer.vocab_size, top_p=1.0).detach()[:,1:])     
        # define a reward for response
        reward, num_errors,num_errors_ref, num_nodes,num_nodes_ref = get_reward(code_ids=response_ids,code_ref_ids=response_ids_ref, tokenizer=tokenizer)
        # reward_ref, num_errors_ref, num_nodes_ref = get_reward(code_ids=response_ids_ref,code_ref_ids=response_ids, tokenizer=tokenizer)
        epoch_rewards += sum([sum(reward.tolist()[i]) for i in range(reward.shape[0])])
        epoch_nerrors += sum(num_errors)
        epoch_nnodes += sum(num_nodes)
        epoch_nerrors_ref += sum(num_errors_ref)
        epoch_nnodes_ref += sum(num_nodes_ref)
        epoch_seen += len(source_ids)

        pbar.set_description('Avg # errors per sample:'+str(round(epoch_nerrors/epoch_seen, 5)))

        # train model with ppo
        train_stats = ppo_trainer.step(source_ids, source_mask, response_ids,response_ids_ref, reward.to(args.device))
        # print(train_stats)
        mean_kl = train_stats['objective/kl']
        mean_entropy = train_stats['objective/entropy']
        # kl = train_stats['ppo/policy/policykl']
        # print(kl.shape, kl.mean(axis=-1))

        nsteps += 1
        #save all the results
        direct_path = "/home/grads/parshinshojaee/trl_code/trl_code/"
        # with open(direct_path + 'results/baselines.csv', 'a') as f:
        # with open(direct_path + 'results/codet5_ppo_steps.csv', 'a') as f:
        with open(direct_path + 'results/codet5_ppo_steps2.csv', 'a') as f:
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
                    ',' + str(epoch_rewards)+
                    ',' + str(round(epoch_rewards/epoch_seen, 4))+
                    ',' + str(epoch_nerrors) +
                    ',' + str(round(epoch_nerrors/epoch_seen, 4))+
                    ',' + str(epoch_nerrors_ref) +
                    ',' + str(round(epoch_nerrors_ref/epoch_seen, 4))+
                    ',' + str(epoch_nnodes) +
                    ',' + str(round(epoch_nnodes/epoch_seen, 4))+
                    ',' + str(epoch_nnodes_ref) +
                    ',' + str(round(epoch_nnodes_ref/epoch_seen,4)) +
                    ',' + str(mean_kl) +
                    ',' + str(mean_entropy)
                    + '\n')


    #plot the reward and nerrors over steps   
    # compute bleu
    nerrors, nerrors_ref = test(ep,train_features, train_dataloader, 'train')
      
    # print ('epoch', ep, '# errors', epoch_nerrors, 'BLEU', str(round(bleu,2))+'/'+str(round(bleu_ref,2)), \
    #       'nerrors', str(nerrors)+'/'+str(nerrors_ref))

    print ('epoch', ep, '# errors', epoch_nerrors, 'nerrors', str(nerrors)+'/'+str(nerrors_ref))
