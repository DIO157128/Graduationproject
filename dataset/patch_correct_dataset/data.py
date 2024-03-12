import json
import os

import javalang
import tqdm
from javalang.ast import Node
from anytree import AnyNode
def read_json(file_path):
    codes = []
    doc_strings = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = js['code']
            doc_string = js['docstring']
            codes.append(code)
            doc_strings.append(doc_string)

    return codes,doc_strings
def get_token(node):
    token = ''
    #print(isinstance(node, Node))
    #print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)
def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)
def create_graph(asts,vocabdict):
    pathlist = []
    treelist = []
    for tree in asts:
        #print(tree)
        #print(path)
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)
        #print(path)
        #print(newtree)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)

        #x = torch.tensor(x, dtype=torch.long, device=device)
        edge_index=[edgesrc, edgetgt]
        #edge_index = torch.tensor([edgesrc, edgetgt], dtype=torch.long, device=device)
        astlength=len(x)
        #print(x)
        #print(edge_index)
        #print(edge_attr)
        treelist.append([[x,edge_index],astlength])
    return treelist
def write_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')

def parseAST(programtext,alltokens):
    programtokens = javalang.tokenizer.tokenize(programtext.strip())
    # print(list(programtokens))
    parser = javalang.parse.Parser(programtokens)
    programast = parser.parse_member_declaration()
    get_sequence(programast, alltokens)
    return programast
def createFinetune():
    directorys = ['./methods/CORRECT','./methods/INCORRECT']
    pairs = []
    tem_pair = []
    cur_label = 1
    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if len(tem_pair)==2:
                    tem_pair.append(cur_label)
                    pairs.append(tem_pair)
                    tem_pair = []
                file_path = os.path.join(root, file)
                f= open(file_path,'r',encoding='utf-8')
                content = f.read()
                tem_pair.append(content)
        cur_label=-1
    if len(tem_pair) == 2:
        tem_pair.append(cur_label)
        pairs.append(tem_pair)
        tem_pair = []
    alltokens = []
    count = 0
    pairs_with_ast = []
    for p in tqdm.tqdm(pairs):
        code1 = p[0]
        code2 = p[1]
        label = p[2]
        try:
            ast1 = parseAST(code1,alltokens)
            ast2 = parseAST(code2,alltokens)
            pairs_with_ast.append([code1,code2,ast1,ast2,label])
        except Exception:
            count+=1
            continue
    print(count)
    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    print(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    pairs_with_ast_tokenized = []
    for p in tqdm.tqdm(pairs_with_ast):
        code1 = p[0]
        code2 = p[1]
        ast1 = p[2]
        ast2 = p[3]
        label = p[4]
        trees = create_graph([ast1,ast2],vocabdict)
        pairs_with_ast_tokenized.append([code1,code2,trees[0],trees[1],label])
    res = [{'code1': p[0], 'code2': p[1], 'ast1': p[2], 'ast2': p[3], 'label': p[4]} for p in
           pairs_with_ast_tokenized]
    write_jsonl('./all.jsonl',res)
if __name__ == '__main__':
    createFinetune()
