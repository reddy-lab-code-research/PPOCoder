import numpy as np
from tree_sitter import Language, Parser
import re
import torch
from code_prepro.lang_processors import *
from compiler.terminal_compiler import TerminalCompiler
import sys
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
sys.path.insert(0, '/home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/CodeBLEU/')
from calc_code_bleu import calc_code_bleu





code_tokenizers = {"java": java_tokenizer, "cpp": cpp_tokenizer, "c": c_tokenizer, "python": py_tokenizer,
                   "javascript": js_tokenizer, "php": php_tokenizer, "c_sharp": cs_tokenizer}
code_detokenizers = {"java": java_detokenizer, "cpp": cpp_detokenizer, "c": c_detokenizer, "python": py_detokenizer,
                   "javascript": js_detokenizer, "php": php_detokenizer, "c_sharp": cs_detokenizer}

lang2compiler = {
    "python": TerminalCompiler("Python"),
    "java": TerminalCompiler("Java"),
    "cpp": TerminalCompiler("C++"),
    "c_sharp": TerminalCompiler("C#"),
    "c": TerminalCompiler("C"),
    "php": TerminalCompiler("PHP"),
}

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


def get_reward(lang, code_ids=None,code_ref_ids=None,gold_ids=None, tokenizer=None):
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
    codes_gold = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(gold_ids, eos_positions)] 
        
    codes = [code_detokenizers[lang](code) for code in codes]
    
    compilation = [lang2compiler[lang].compile_code_string(code) for code in codes]

    codes = [remove_special_tokens(code) for code in codes]
    codes_ref = [remove_special_tokens(code) for code in codes_ref]
    codes_gold = [remove_special_tokens(code) for code in codes_gold]
    error_node_counts = [tree_sitter_full_compile(code,lang) for code in codes]
    error_node_counts_ref = [tree_sitter_full_compile(code,lang) for code in codes_ref]
    error_node_counts_gold = [tree_sitter_full_compile(code,lang) for code in codes_gold]
    num_errors = [i[0] for i in error_node_counts]
    num_errors_ref = [i[0] for i in error_node_counts_ref]  
    num_errors_gold = [i[0] for i in error_node_counts_gold]  
    num_nodes = [i[1] for i in error_node_counts]
    num_nodes_ref = [i[1] for i in error_node_counts_ref]
    num_nodes_gold = [i[1] for i in error_node_counts_gold]
    
    keywords_dir = 'CodeBLEU/keywords/'
    # ast_match = calc_code_bleu([codes_gold], codes, lang, keywords_dir)[2]
    # dfg_match = calc_code_bleu([codes_gold], codes, lang, keywords_dir)[3]
    
    rewards = np.zeros_like(code_ids, dtype=np.float)
    ast_match_batch = 0
    dfg_match_batch = 0
    compile_batch = 0
    for i in range(len(rewards)):
        _, _, did_compile = compilation[i]
        reward = 1 if did_compile else -1
        
        ast_match = calc_code_bleu([[codes_gold[i]]], [codes[i]], lang, keywords_dir)[2]
        dfg_match = calc_code_bleu([[codes_gold[i]]], [codes[i]], lang, keywords_dir)[3]

        rewards[i, min(eos_positions[i],max_len-1)] = reward + ast_match + dfg_match
        compile_batch += reward
        ast_match_batch += ast_match
        dfg_match_batch += dfg_match
     
    mean_rate = compile_batch/len(codes)
    mean_ast_match =  ast_match_batch/len(codes) 
    mean_dfg_match =  dfg_match_batch/len(codes)  
    return torch.Tensor(rewards),mean_rate,mean_ast_match,mean_dfg_match, num_errors, num_errors_ref, num_nodes, num_nodes_ref

