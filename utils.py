import json
import os

import javalang
import tqdm
from javalang.ast import Node
from anytree import AnyNode
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

def create_tokenized_ast(asts,vocabdict):
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


def getvocabdict(all_codes):
    alltokens = []
    count = 0
    for code in tqdm.tqdm(all_codes):
        try:
            programtokens = javalang.tokenizer.tokenize(code)
            # print(list(programtokens))
            parser = javalang.parse.Parser(programtokens)
            programast = parser.parse_member_declaration()
            get_sequence(programast, alltokens)
        except Exception:
            count+=1
    print("failed:{}".format(count))
    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    print("vocabsize:{}".format(vocabsize))
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    return vocabdict

def read_json(js_path):
    codes = []
    docs = []
    with open(js_path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            js = json.loads(line)
            codes.append(js['code'])
            docs.append(js['docstring'])
    return codes,docs

def read_abs_json(js_path):
    codes = []
    docs = []
    abs_codes = []
    with open(js_path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            js = json.loads(line)
            codes.append(js['code'])
            docs.append(js['docstring'])
            abs_codes.append(js['abs_code'])
    return codes,docs

def write_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')
if __name__ == '__main__':
    read_json('./dataset/pretrain_dataset/original_data/test.jsonl')