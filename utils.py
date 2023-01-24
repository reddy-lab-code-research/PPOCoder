
__all__ = ['extract_structure','flatten_dict', 'stack_dicts', 'add_suffix', 'pad_to_size', 'logprobs_from_logits', 'whiten',
           'clip_by_value', 'entropy_from_logits', 'average_torch_dicts', 'stats_to_np', 'build_bert_batch_from_txt']

# Cell
import torch
import torch.nn.functional as F
import collections
import numpy as np
from tqdm import tqdm
from parser import (tree_to_token_index,
                   tree_to_token_nodes,
                   index_to_code_token,
                   tree_to_variable_index, 
                   detokenize_code)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from tree_sitter import Language, Parser
import pickle

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
                if (args.l1=='php') and not(line1.startswith('<?php')):
                    line1 = '<?php '+line1
                if (args.l2 =='php') and not(line2.startswith('<?php')):
                    line2 = '<?php '+line2
                    
                orig_line1, orig_line2 = line1, line2
                
                if args.l1=='python':
                    line1 = detokenize_code(line1)
                else:
                    line1 = line1.replace('NEW_LINE', '\n')
                if args.l2=='python':
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


def extract_structure(code, parser, lang):  
    try:
        # ast
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        ast_token_nodes = tree_to_token_nodes(root_node)
        tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index] 
        
        # dfg
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg,ast_token_nodes


def get_lr_path(leaf):
    if leaf==-1:
        return -1
    path = [leaf]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path


def get_node_types(node, l):
    l.append(node.type)
    for child in node.children:
        get_node_types(child, l)
        
        
def gather_node_types(examples, args):
    global node_types
    filename = args.output_dir+'/node_types.pkl'
    node_types = []
    for example in tqdm(examples):
        root = parsers[args.source_lang][0].parse(bytes(example.source,'utf8')).root_node 
        get_node_types(root, node_types)
        root = parsers[args.target_lang][0].parse(bytes(example.target,'utf8')).root_node 
        get_node_types(root, node_types)
    node_types = sorted(list(set(node_types)))
    pickle.dump(node_types, open(filename, 'wb'))
    node_types = {t:i for i,t in enumerate(node_types)}


def convert_path_to_idx(path, max_depth):
    if path==-1:
        return [-1]*max_depth
    path = [node_types.get(node.type, -1) for node in path][:max_depth]
    path = path + [-1]*(max_depth-len(path))
    return path


def convert_examples_to_ast_dfg(examples, tokenizer, args, stage=None):
    features = []
    match, nomatch = 1,1
    smatch, snomatch = 1,1
    bar = tqdm(enumerate(examples))
    for example_index, example in bar: 
        target_tokens = tokenizer.tokenize(example.target_orig)[:args.max_source_length-2]
        code_tokens,dfg,ast = extract_structure(example.target, parsers[args.target_lang], args.target_lang)
        for i in range(1, len(ast)):
            if (ast[i].start_point[0]<ast[i-1].start_point[0]) or \
                    ((ast[i].start_point[0]==ast[i-1].start_point[0]) and (ast[i].start_point[1]<ast[i-1].start_point[1])):
                raise Exception("Leaves not ordered by position in sequence.")      
    tcode = list(''.join(target_tokens).replace('Ġ', ' ').replace('ĉ', '\t'))
    scode = list(''.join(code_tokens))
    tcode_to_scode = []
    j = 0
    for i in range(len(tcode)):
        if j<len(scode):
            if tcode[i]==scode[j]:
                tcode_to_scode.append(j)
                j += 1
                match += 1
            else:
                tcode_to_scode.append(-1)
                if (tcode[i]!=' '):
                    if (tcode[i] not in [' ','N','E','W','_','L','I','N','E']):
                        nomatch += 1
        else:
            tcode_to_scode.append(-1)
            if (tcode[i]!=' '):
                if (tcode[i] not in [' ','N','E','W','_','L','I','N','E']):
                    nomatch += 1
    tcode_to_target = []
    for i in range(len(target_tokens)):
        tcode_to_target += [i]*len(target_tokens[i])
    scode_to_code = []
    for i in range(len(code_tokens)):
        scode_to_code += [i]*len(code_tokens[i])  
    target_to_code = [[] for i in range(len(target_tokens))]
    for i in range(len(tcode)):
        if tcode_to_scode[i]>=0:
            target_to_code[tcode_to_target[i]].append( scode_to_code[tcode_to_scode[i]] )
    code_to_target = [[] for i in range(len(code_tokens))]
    for i in range(len(target_to_code)):
        for c in set(target_to_code[i]):
            code_to_target[c].append(i) 

    target_tokens = target_tokens+[tokenizer.sep_token]            
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    target_len = len(target_ids)
    target_dfg = np.zeros((target_len, target_len))
    target_ast = -np.ones((target_len, args.max_ast_depth))
    target_ast_sim = -np.ones((target_len, target_len))
    tlr_paths = [get_lr_path(leaf) for leaf in ast]
    tlr_paths = [convert_path_to_idx(path, args.max_ast_depth) for path in tlr_paths]
    for i,ts in enumerate(code_to_target):
        target_ast[ts, :] = np.array(tlr_paths[i]).reshape((1,-1))
    for _,l,_,_,rs in dfg:
        for lt in code_to_target[l]:
            for r in rs:
                target_dfg[lt, code_to_target[r]] = 1
    target_dfg[-1,:] = -1
    target_dfg[:,-1] = -1
    
    return target_dfg, target_ast


def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat

def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        max_len = max([len(l) for l in stats_list])
        stats_list = [torch.cat((l.cpu(),torch.ones(max_len-len(l)))) for l in stats_list]
        results[k] = torch.stack(stats_list)
    return results

def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k,v in input_dict.items())

# Cell

def pad_to_size(tensor, size, dim=1, padding=50256):
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size==size:
        return tensor
    else:
        return torch.nn.functional.pad(tensor, (0,size-t_size), 'constant', padding)

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy
    
    # logpy = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(-1)
    # logp = F.log_softmax(logpy, dim=-1)
    # return logp


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    var = torch.nan_to_num(var, nan=1.0)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd*logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    """Average values of a list of dicts wiht torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict

def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


# Cell

def build_bert_batch_from_txt(text_list, tokenizer, device):
    """Create token id and attention mask tensors from text list for BERT classification."""

    # tokenize
    tensors = [tokenizer.encode(txt, return_tensors="pt").to(device) for txt in text_list]

    # find max length to pad to
    max_len = max([t.size()[1] for t in tensors])

    # get padded tensors and attention masks
    # (attention masks make bert ignore padding)
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        attention_mask = torch.ones(tensor.size(), device=device)
        padded_tensors.append(pad_to_size(tensor, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_mask, max_len, padding=0))

    # stack all tensors
    padded_tensors = torch.cat(padded_tensors)
    attention_masks = torch.cat(attention_masks)

    return padded_tensors, attention_masks

