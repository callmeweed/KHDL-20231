import torch
import torch.nn as nn
import torch.nn.functional as F
from config import features_config

class CategoricalEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, input_type="single"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_type = input_type
        if self.input_type == "multi":
            self.W = nn.Linear(d_model, 1)
            self.ff = nn.Linear(d_model, d_model)
        else:
            self.ff = nn.Linear(d_model, d_model)
    
    def forward(self, input_ids, att_mask=None):
        embedded = self.embedding(input_ids) # B, N, d_model
        if self.input_type == "multi":
            att = self.W(embedded).reshape(att_mask.size())
            if att_mask is not None:
                att = att.masked_fill_(att_mask==0, -float('inf'))
            att_weights = F.softmax(att, dim=-1)
            return F.relu(torch.mul(self.ff(embedded), att_weights.unsqueeze(-1)).sum(dim=-2))
        else:
            return F.relu(self.ff(embedded))[:,0,:]
        
    
class RevenuePredictor(nn.Module):
    def __init__(self, roberta, config):
        super().__init__()
        
        d_model = config["d_model"]
        
        # text encoder
        self.text_embedding = roberta.embeddings
        self.text_encoder = roberta.encoder.layer[:config["text"]["num_encoder"]]
        self.adapt = nn.Linear(config["text"]["d_model"], d_model)
        self.get_extended_attention_mask = roberta.get_extended_attention_mask
        
        # categorical encoders
        scat_cf = config["categorical"]["single"]
        self.country_encoder = CategoricalEncoder(scat_cf["country"]["vocab_size"], d_model, input_type="single")
        self.language_encoder = CategoricalEncoder(scat_cf["language"]["vocab_size"], d_model, input_type="single")
        self.start_year_encoder = CategoricalEncoder(scat_cf["start_year"]["vocab_size"], d_model, input_type="single")
        self.crew_encoder = CategoricalEncoder(scat_cf["crew"]["vocab_size"], d_model, input_type="single")
        self.writer_encoder = CategoricalEncoder(scat_cf["writer"]["vocab_size"], d_model, input_type="single")
        
        mcat_cf = config["categorical"]["multi"]
        self.genre_encoder = CategoricalEncoder(mcat_cf["genre"]["vocab_size"], d_model, input_type="multi")
        self.company_encoder = CategoricalEncoder(mcat_cf["company"]["vocab_size"], d_model, input_type="multi")
        self.cast_encoder = CategoricalEncoder(mcat_cf["cast"]["vocab_size"], d_model, input_type="multi")
        
        # budget encoder
        self.budget_encoder = nn.Linear(1, d_model)
        self.runtime_encoder = nn.Linear(1, d_model)
        
        # self attention
        self.W = nn.Linear(d_model, 1)
        self.ff = nn.Linear(d_model, d_model)
        
        # feed forward
        self.fc_out = nn.Linear(d_model, 1)
        self.ff_dropout = nn.Dropout(0.1)
        
#     def forward(self, col, lan, rel, crew, adult, gen, gen_mask, com, com_mask, cast, cast_mask, bud, text, text_mask):
    def forward(self, batch):
        """
        col, lan, rel, crew: B
        gen, gen_mask, com, com_mask, cast, cast_mask: BxN_i
        bud: Bx1
        text, text_mask: BxN_j
        """
        o_cou = self.country_encoder(batch["cou"])
        o_lan = self.language_encoder(batch["lan"])
        o_sta = self.start_year_encoder(batch["sta"])
        o_crew = self.crew_encoder(batch["crew"])
        o_wri = self.writer_encoder(batch["wri"])
        o_gen = self.genre_encoder(batch["gen"], batch["gen_mask"])
        o_com = self.company_encoder(batch["com"], batch["com_mask"])
        o_cast = self.cast_encoder(batch["cast"], batch["cast_mask"])
        o_bud = F.relu(self.budget_encoder(batch["bud"]))
        o_run = F.relu(self.runtime_encoder(batch["run"]))
        o_text = self.forward_text_encoder(batch["text"], batch["text_mask"])
#         for x in [o_col, o_lan, o_rel, o_crew, o_gen, o_adult, o_com, o_cast, o_bud, o_text]:
#             print(x.shape)
        
        concat = torch.cat([x.unsqueeze(1) for x in [o_cou, o_lan, o_sta, o_crew, o_wri, o_gen, o_com, o_cast, o_bud, o_run , o_text]], dim=-2)
#         print(o_cou.shape)
#         print(o_gen.shape)
#         print(o_bud.shape)
#         print(o_text.shape)
#         print(concat.shape)
        hidden, att_weights = self.self_att(concat)
        hidden = self.ff_dropout(hidden)
        return F.tanh(self.fc_out(hidden)).reshape(hidden.shape[0],), att_weights
        
    def forward_text_encoder(self, text_ids, att_mask):
        max_seq_length = att_mask.sum(dim=-1).max().item()
        text_ids = text_ids[:, :max_seq_length]
        att_mask = att_mask[:, :max_seq_length]
        batch_size, seq_length = text_ids.size()
        
        device = text_ids.device
        if hasattr(self.text_embedding, "token_type_ids"):
            buffered_token_type_ids = self.text_embedding.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(text_ids.size(), dtype=torch.long, device=device)
        hidden_state = self.text_embedding(input_ids=text_ids, token_type_ids=token_type_ids)
        
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(att_mask, text_ids.size())
        
        for layer in self.text_encoder:
            hidden_state = layer(hidden_state, extended_attention_mask)[0]
        return F.relu(self.adapt(hidden_state[:, 0, :]))
    
    def self_att(self, x):
        att = self.W(x)
#         print(att.shape)
        att_weights = F.softmax(att, dim=-1)
#         print(self.ff(x).shape)
#         print(att_weights.unsqueeze(-1).shape)
        return torch.mul(self.ff(x), att_weights).sum(dim=-2), att_weights