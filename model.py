import numpy as np
from transformers import T5ForConditionalGeneration, T5Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv

from torch_geometric.nn.glob import GlobalAttention
class GGNN(torch.nn.Module):
    def __init__(self,vocablen,embedding_dim,num_layers,device):
        super(GGNN, self).__init__()
        self.device=device
        #self.num_layers=num_layers
        self.embed=nn.Embedding(vocablen,embedding_dim)
        self.edge_embed=nn.Embedding(20,embedding_dim)
        #self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) for i in range(num_layers)])
        self.ggnnlayer=GatedGraphConv(embedding_dim,num_layers)
        self.mlp_gate=nn.Sequential(nn.Linear(embedding_dim,1),nn.Sigmoid())
        self.pool=GlobalAttention(gate_nn=self.mlp_gate)

    def forward(self, data):
        res_vector = []
        for d in data:
            x, edge_index = d
            x = torch.tensor(x,dtype=torch.long,device=self.device)
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            x = self.embed(x)
            x = x.squeeze(1)
            x = self.ggnnlayer(x, edge_index)
            batch=torch.zeros(x.size(0),dtype=torch.long).to(self.device)
            hg=self.pool(x,batch=batch)
            res_vector.append(hg[0])
        res_vector = torch.stack(res_vector, dim=0)
        return res_vector
def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss

class ConcatenateModel(torch.nn.Module):
    def __init__(self, args):
        super(ConcatenateModel, self).__init__()
        self.args = args
        self.seq_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(args.device)
        self.graph_model = GGNN(854,768,4,args.device).to(args.device)
        self.eos_id = T5Config.from_pretrained("Salesforce/codet5-base").eos_token_id
    def encode_text(self,input_ids,attention_mask):
        outputs = self.seq_model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=False)
        hidden_states = outputs['encoder_last_hidden_state']
        eos_mask = input_ids.eq(self.eos_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec
    def forward(self, input_ids, attention_mask,asts):
        s_image_features = self.graph_model(asts)
        s_text_features = self.encode_text(input_ids, attention_mask)

        # normalized features
        s_image_features = s_image_features / s_image_features.norm(dim=-1, keepdim=True)
        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits

        labels = torch.arange(s_image_features.shape[0]).to(self.args.device)
        node_loss = cal_cl_loss(s_image_features, s_text_features, labels)

        all_loss = node_loss
        return all_loss
class CloneDetectionConcatenateModel(torch.nn.Module):
    def __init__(self, args):
        super(CloneDetectionConcatenateModel, self).__init__()
        self.args = args
        self.seq_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(args.device)
        self.graph_model = GGNN(854,768,4,args.device).to(args.device)
        self.eos_id = T5Config.from_pretrained("Salesforce/codet5-base").eos_token_id
    def encode_text(self,input_ids,attention_mask):
        outputs = self.seq_model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=False)
        hidden_states = outputs['encoder_last_hidden_state']
        eos_mask = input_ids.eq(self.eos_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec
    def forward(self, code1_ids, code1_ne, code2_ids, code2_ne, label,ast1,ast2,test=False):
        s1_image_features = self.graph_model(ast1)
        s1_text_features = self.encode_text(code1_ids, code1_ne)
        s2_image_features = self.graph_model(ast2)
        s2_text_features = self.encode_text(code2_ids, code2_ne)
        # normalized features
        s1_image_features = s1_image_features / s1_image_features.norm(dim=-1, keepdim=True)
        s1_text_features = s1_text_features / s1_text_features.norm(dim=-1, keepdim=True)
        s2_image_features = s2_image_features / s2_image_features.norm(dim=-1, keepdim=True)
        s2_text_features = s2_text_features / s2_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        s1_tensor = torch.cat((s1_image_features,s1_text_features),dim=1)
        s2_tensor = torch.cat((s2_image_features, s2_text_features), dim=1)
        sim = F.cosine_similarity(s1_tensor,s2_tensor)
        if test:
            return sim
        else:
            final_loss = F.mse_loss(sim,label)
            return final_loss
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768*2, 768*2)
        self.out_proj = nn.Linear(768*2, 2)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x
class ClassificationConcatenateModel(torch.nn.Module):
    def __init__(self, args):
        super(ClassificationConcatenateModel, self).__init__()
        self.args = args
        self.seq_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(args.device)
        self.graph_model = GGNN(854,768,4,args.device).to(args.device)
        self.eos_id = T5Config.from_pretrained("Salesforce/codet5-base").eos_token_id
        self.classifier = RobertaClassificationHead().to(args.device)
    def encode_text(self,input_ids,attention_mask):
        outputs = self.seq_model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=False)
        hidden_states = outputs['encoder_last_hidden_state']
        eos_mask = input_ids.eq(self.eos_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec
    def forward(self, code1_ids, code1_ne, label,ast1,test=False):
        s1_image_features = self.graph_model(ast1)
        s1_text_features = self.encode_text(code1_ids, code1_ne)
        # normalized features
        s1_image_features = s1_image_features / s1_image_features.norm(dim=-1, keepdim=True)
        s1_text_features = s1_text_features / s1_text_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        vec = torch.cat((s1_image_features,s1_text_features),dim=1)
        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if test:
            return prob
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label)
            return loss, prob
class APCAConcatenateModel(torch.nn.Module):
    def __init__(self, args):
        super(APCAConcatenateModel, self).__init__()
        self.args = args
        self.seq_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(args.device)
        self.graph_model = GGNN(854,768,4,args.device).to(args.device)
        self.eos_id = T5Config.from_pretrained("Salesforce/codet5-base").eos_token_id
        self.classifier = RobertaClassificationHead().to(args.device)
    def encode_text(self,input_ids,attention_mask):
        outputs = self.seq_model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=False)
        hidden_states = outputs['encoder_last_hidden_state']
        eos_mask = input_ids.eq(self.eos_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec
    def forward(self, code1_ids, code1_ne, code2_ids, code2_ne,label,ast1,ast2,test=False):
        s1_image_features = self.graph_model(ast1)
        s1_text_features = self.encode_text(code1_ids, code1_ne)
        s2_image_features = self.graph_model(ast2)
        s2_text_features = self.encode_text(code2_ids, code2_ne)
        # normalized features
        s1_image_features = s1_image_features / s1_image_features.norm(dim=-1, keepdim=True)
        s1_text_features = s1_text_features / s1_text_features.norm(dim=-1, keepdim=True)
        s2_image_features = s2_image_features / s2_image_features.norm(dim=-1, keepdim=True)
        s2_text_features = s2_text_features / s2_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        s1_tensor = torch.cat((s1_image_features, s1_text_features), dim=1)
        s2_tensor = torch.cat((s2_image_features, s2_text_features), dim=1)
        vec = torch.cat((s1_tensor, s2_tensor), dim=1)
        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if test:
            return prob
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label)
            return loss, prob
