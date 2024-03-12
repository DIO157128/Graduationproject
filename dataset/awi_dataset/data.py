import json
import os

import javalang
import pandas as pd
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
        treelist.append([[x,edge_index,edge_attr],astlength])
    return treelist
def write_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')
def get_all_data():
    file_paths = ['./original_data/awi_train.csv', './original_data/awi_val.csv', './original_data/awi_test.csv']
    saved_file_paths = ['./awi_train.jsonl', './awi_val.jsonl', './awi_test.jsonl']

    ##get vocab_dict
    all_codes = []
    alltokens = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        codes = df['source'].tolist()
        all_codes.extend(codes)
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
    print(count)
    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    for file_path,saved_file_path in zip(file_paths,saved_file_paths):
        df = pd.read_csv(file_path)
        codes = df['source'].tolist()
        labels = df['target'].tolist()

        final_codes = []
        final_labels = []
        final_asts = []
        count = 0
        for idx in tqdm.tqdm(range(len(codes))):
            try:
                code = codes[idx]
                programtokens = javalang.tokenizer.tokenize(code)
                # print(list(programtokens))
                parser = javalang.parse.Parser(programtokens)
                programast = parser.parse_member_declaration()
                final_asts.append(programast)
                final_codes.append(code)
                final_labels.append(labels[idx])
            except Exception:
                count+=1
                continue
        print(count)
        treedict = create_graph(final_asts, vocabdict)
        res = [{'code':c,'label':l,'ast':a}for c,l,a in zip(final_codes,final_labels,treedict)]
        write_jsonl(saved_file_path,res)

def split(file_path,towrite):
    res = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            res.append(js)
            if idx>100:
                break
    write_jsonl(towrite,res)


if __name__ == '__main__':
    # get_all_data()
    # split('awi_train.jsonl', 'awi_train_part.jsonl')
    split('awi_val.jsonl', 'awi_val_part.jsonl')
    split('awi_test.jsonl', 'awi_test_part.jsonl')


