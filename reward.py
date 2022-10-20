import numpy as np
from tree_sitter import Language, Parser
import re
import torch

parsers={}        
for lang in ['python']:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)   
    parsers[lang]= parser
    
def remove_special_tokens(code_string):
    lines = code_string.split("NEW_LINE")
    lines = [item.strip() for item in lines]
    
    curr_indent = 0
    new_lines = []
    for line in lines:
        indent_count = line.count('INDENT')
        dedent_count = line.count('DEDENT')
        curr_indent += indent_count - dedent_count
        wo_indent = re.sub('INDENT\s?', '', line)
        wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
        new_lines.append('\t'*curr_indent + wo_dedent)
    return ("\n").join(new_lines)

def dfs_parse_tree(node, level, count_list, verbose = False):
    if verbose:
        if node.type == 'ERROR':
            print (level, '-'*(level*2), colored(node.type, 'red'))
        else:
            print (level, '-'*(level*2), node.type)
    if node.type == 'ERROR':
        count_list[0]+=1
    else:
        count_list[1]+=1
    for child in node.children:
        dfs_parse_tree(child, level+1, count_list, verbose)
    return

def tree_sitter_full_compile(code, lang='python', verbose = False):
    root=parsers[lang].parse(bytes(code, 'utf-8')).root_node
    count_list = [0, 0]
    dfs_parse_tree(root, 0, count_list, verbose)
    return count_list


# def get_reward(code_ids=None, tokenizer=None):
#     code_ids = np.array(code_ids.cpu())
#     eos_positions = [(id==tokenizer.eos_token_id).argmax() for id in code_ids]
#     codes = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
#              for id,eos_pos in zip(code_ids, eos_positions)]
        
#     codes = [remove_special_tokens(code) for code in codes]
#     error_node_counts = [tree_sitter_full_compile(code) for code in codes]
#     num_errors = [i[0] for i in error_node_counts]
#     num_nodes = [i[1] for i in error_node_counts]
#     rewards = np.zeros_like(code_ids)
#     for i in range(len(rewards)):
#         rewards[i, eos_positions[i]] = num_nodes[i]-num_errors[i]
#     return torch.Tensor(rewards), num_errors, num_nodes




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
    
    rewards = np.zeros_like(code_ids, dtype=np.float)
    for i in range(len(rewards)):
        # rewards[i, min(eos_positions[i],max_len-1)] = (num_nodes[i]-num_errors[i])*0.001 
        # rewards[i, min(eos_positions[i],max_len-1)] = (num_nodes[i]-num_errors[i])
        rewards[i, min(eos_positions[i],max_len-1)] = (num_nodes[i]-args.loss_W*num_errors[i])*args.reward_W
        # rewards[i, min(eos_positions[i],max_len-1)] = -num_errors[i]
    
    ###############################
    #MODIFED PARSHIN:  
    rewards = np.zeros_like(code_ids, dtype=np.float)
    # breakpoint()
    for i in range(len(rewards)):
        rewards[i, min(eos_positions[i],max_len-1)] = -args.reward_W*(num_nodes[i] - num_nodes_ref[i])**2
    ###############################
    return torch.Tensor(rewards), num_errors,num_errors_ref, num_nodes, num_nodes_ref
    
    
    
    
    
    
    
    
    
    
    