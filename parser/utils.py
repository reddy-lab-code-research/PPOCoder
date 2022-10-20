import re
from io import StringIO
import  tokenize
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
                
        code = '\n'.join(temp)
        pos = 0
        while pos<len(code):
            try:
                start = code[pos:].index('"""') + pos
                end = code[start+3:].index('"""') + start+3
                code = code[:start]+code[end+3:]
                pos = start
            except:
                break
        return re.sub(r"\r\n\s*\r\n",'\n',code)
    
    
    
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def tree_to_token_index(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens
    
def tree_to_token_nodes(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [root_node]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_nodes(child)
        return code_tokens
    
def tree_to_variable_index(root_node,index_to_code):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        index=(root_node.start_point,root_node.end_point)
        _,code=index_to_code[index]
        if root_node.type!=code:
            return [(root_node.start_point,root_node.end_point)]
        else:
            return []
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_variable_index(child,index_to_code)
        return code_tokens    

def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s
   
def detokenize_code(code):
        # replace recreate lines with \n and appropriate indent / dedent
        # removing indent/ dedent tokens
        assert isinstance(code, str) or isinstance(code, list)
        if isinstance(code, list):
            code = " ".join(code)
        code = code.replace("ENDCOM", "NEW_LINE")
        code = code.replace("▁", "SPACETOKEN")
        lines = code.split("NEW_LINE")
        tabs = ""
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("INDENT "):
                tabs += "    "
                line = line.replace("INDENT ", tabs)
            elif line.startswith("DEDENT"):
                number_dedent = line.count("DEDENT")
                tabs = tabs[4 * number_dedent :]
                line = line.replace("DEDENT", "")
                line = line.strip()
                line = tabs + line
            elif line == "DEDENT":
                line = ""
            else:
                line = tabs + line
            lines[i] = line
        untok_s = "\n".join(lines)
        # find string and comment with parser and detokenize string correctly
        try:
            for toktype, tok, _, _, line in tokenize.tokenize(
                BytesIO(untok_s.encode("utf-8")).readline
            ):
                if toktype == tokenize.STRING or toktype == tokenize.COMMENT:
                    tok_ = (
                        tok.replace("STRNEWLINE", "\n")
                        .replace("TABSYMBOL", "\t")
                        .replace(" ", "")
                        .replace("SPACETOKEN", " ")
                    )
                    untok_s = untok_s.replace(tok, tok_)
        except KeyboardInterrupt:
            raise
        except:
            # TODO raise ValueError(f'Invalid python function \n {code}\n')
            pass
#         # detokenize imports
#         untok_s = (
#             untok_s.replace(". ", ".")
#             .replace(" .", ".")
#             .replace("import.", "import .")
#             .replace("from.", "from .")
#         )
#         # special strings
#         string_modifiers = ["r", "u", "f", "rf", "fr", "b", "rb", "br"]
#         for modifier in string_modifiers + [s.upper() for s in string_modifiers]:
#             untok_s = untok_s.replace(f" {modifier} '", f" {modifier}'").replace(
#                 f' {modifier} "', f' {modifier}"'
#             )
#         untok_s = untok_s.replace("> >", ">>").replace("< <", "<<")
        return untok_s