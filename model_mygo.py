import torch
import torch.nn as nn

from model_new import *

class MyGO(nn.Module):
    def __init__(
            self, 
            num_ent, 
            num_rel, 
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout = 0.1,
            emb_dropout = 0.6, 
            vis_dropout = 0.1, 
            txt_dropout = 0.1,
            visual_token_index = None, 
            text_token_index = None,
            score_function = "tucker",
            text_tokenizer = "bert",
            visual_tokenizer = "beit",
            adv_generator = None
        ):
        super(MyGO, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        # Load Different Tokenizers
        if text_tokenizer == "bert":
            textual_tokens = torch.load("tokens/textual.pth")
        elif text_tokenizer == "roberta":
            textual_tokens = torch.load("tokens/textual_roberta.pth")
        elif text_tokenizer == "llama":
            textual_tokens = torch.load("tokens/textual_llama.pth")
        else:
            raise NotImplementedError
        if visual_tokenizer == "beit":
            visual_tokens = torch.load("tokens/visual.pth")
        elif visual_tokenizer == "vqgan":
            visual_tokens = torch.load("tokens/visual_vqgan.pth")
        else:
            raise NotImplementedError
        
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.score_function = score_function
        self.img_dim = visual_tokens.shape[1]
        self.txt_dim = textual_tokens.shape[1]

        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        self.false_ents = torch.full((self.num_ent,1),False).cuda()

        self.ent_vis_mask = ent_vis_mask
        self.ent_txt_mask = ent_txt_mask
        #self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim = 1)
        
        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel,1),False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels], dim = 1)
        
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.proj_ent_vis = nn.Linear(self.img_dim, dim_str)
        self.proj_ent_txt = nn.Linear(self.txt_dim, dim_str)
        

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        self.contrastive = ContrastiveLoss(temp=0.5)
        self.num_con = 512
        self.num_vis = ent_vis_mask.shape[1]
        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        else:
            pass
        
        self.init_weights()
        # torch.save(self.visual_token_embedding, open("visual_token.pth", "wb"))
        # torch.save(self.text_token_embedding, open("textual_token.pth", "wb"))

        self.attn_dim_new = int(self.dim_str / self.num_head)
        self.scale = self.attn_dim_new ** -0.5

        self.vis2txt_gate = nn.Sequential(
            nn.Linear(self.dim_str * 2, self.dim_str),
            nn.Sigmoid()
        )
        self.txt2vis_gate = nn.Sequential(
            nn.Linear(self.dim_str * 2, self.dim_str),
            nn.Sigmoid()
        )
        self.ent_attn = nn.Linear(self.dim_str, 1, bias=False)
        self.context_vec = nn.Parameter(torch.randn((1, dim_str)))

        self.generator = adv_generator


    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def ModalityAwareFusion(self, rep_ent_str, rep_ent_vis, rep_ent_txt, ent_vis_mask, ent_txt_mask, p_drop):

        # rep_ent_str = rep_ent_str.repeat(1, rep_ent_vis.size(1), 1)
        # fused_x1 = torch.stack((rep_ent_str, rep_ent_vis, rep_ent_txt), dim=1)
        # u = torch.tanh(fused_x1)
        # fu_scores = self.ent_attn(u).squeeze(-1)
        # attention_weights = torch.softmax(fu_scores, dim=-1)
        # context_vectors = torch.sum(attention_weights.unsqueeze(-1) * fused_x1, dim=1)
        # context_vectors = context_vectors.mean(dim=1, keepdim=True)  # [15000, 256]
        
        #if self.training and torch.rand(1).item() < p_drop:
        if self.training and torch.rand(1).item() < 0.3:
            if torch.rand(1).item() < 0.5:
                rep_ent_txt *= 0
                ent_txt_mask.zero_()
            else:
                rep_ent_vis *= 0
                ent_vis_mask.zero_()
        else:
            vis_avg = rep_ent_vis.mean(1, keepdim=True)  # [B,1,H]?
            txt_avg = rep_ent_txt.mean(1, keepdim=True)
            vis_gate = self.vis2txt_gate(torch.cat([vis_avg, txt_avg], -1))  # [B,1,H]
            txt_gate = self.txt2vis_gate(torch.cat([txt_avg, vis_avg], -1))
            rep_ent_vis = rep_ent_vis * vis_gate
            rep_ent_txt = rep_ent_txt * txt_gate

        # else:
        #     str_avg = rep_ent_str.mean(1, keepdims=True)
        #     vis_avg = rep_ent_vis.mean(1, keepdim=True)  # [B,1,H]?
        #     txt_avg = rep_ent_txt.mean(1, keepdim=True)
        #     vis_gate = self.vis2txt_gate(torch.cat([vis_avg, str_avg], -1))  # [B,1,H]
        #     txt_gate = self.txt2vis_gate(torch.cat([txt_avg, str_avg], -1))
        #     rep_ent_vis = rep_ent_vis * vis_gate
        #     rep_ent_txt = rep_ent_txt * txt_gate

        # vis_avg = rep_ent_vis.mean(1, keepdim=True)  # [B,1,H]?
        # txt_avg = rep_ent_txt.mean(1, keepdim=True)
        # vis_gate = self.vis2txt_gate(torch.cat([vis_avg, txt_avg], -1))  # [B,1,H]
        # txt_gate = self.txt2vis_gate(torch.cat([txt_avg, vis_avg], -1))
        # rep_ent_vis = rep_ent_vis * vis_gate
        # rep_ent_txt = rep_ent_txt * txt_gate
        # if self.training and torch.rand(1).item() < 0.15:
        #     if torch.rand(1).item() < 0.5:
        #         rep_ent_txt *= 0
        #         ent_txt_mask.zero_()
        #     else:
        #         rep_ent_vis *= 0
        #         ent_vis_mask.zero_()
        
        #rep_ent_str = rep_ent_str.repeat(1, rep_ent_vis.size(1), 1)
        #fused_x1 = torch.stack((rep_ent_str, rep_ent_vis, rep_ent_txt), dim=1)
        #u = torch.tanh(fused_x1)
        #fu_scores = self.ent_attn(u).squeeze(-1)
        #attention_weights = torch.softmax(fu_scores, dim=-1)
        #context_vectors = torch.sum(attention_weights.unsqueeze(-1) * fused_x1, dim=1)
        #context_vectors = context_vectors.mean(dim=1, keepdim=True)  # [15000, 256]
        #fused_x = torch.cat([context_vectors, rep_ent_vis, rep_ent_txt], dim = 1)

        #fused_x = torch.cat([rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        fused_mask = torch.cat([ent_vis_mask, ent_txt_mask], dim=1)
        fused_mask = fused_mask == 0

        # str_avg = rep_ent_str.mean(1, keepdims=True)
        # vis_avg = rep_ent_vis.mean(1, keepdim=True)  # [B,1,H]?
        # txt_avg = rep_ent_txt.mean(1, keepdim=True)
        # vis_gate = self.vis2txt_gate(torch.cat([str_avg, vis_avg], -1))  # [B,1,H]
        # txt_gate = self.txt2vis_gate(torch.cat([str_avg, txt_avg], -1))
        # rep_ent_vis = rep_ent_vis * vis_gate
        # rep_ent_txt = rep_ent_txt * txt_gate
        # fused_x = torch.cat([rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        # fused_mask = torch.cat([ent_vis_mask, ent_txt_mask], dim=1)
        # fused_mask = fused_mask == 0

        return rep_ent_str, rep_ent_vis, rep_ent_txt, fused_mask


    def get_multi_attn(self, q, k, v, flag):
        B = q.size(0)
        B2 = q.size(1)
        q = q.view(B, B2, self.num_head, self.attn_dim_new).transpose(1, 2)  # [B, n_heads, 3, head_dim]
        k = k.view(B, B2, self.num_head, self.attn_dim_new).transpose(1, 2)
        v = v.view(B, B2, self.num_head, self.attn_dim_new).transpose(1, 2)

        scores_new = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_new = torch.softmax(scores_new, dim=-1)

        #attn_new = self.dropout_out_new(attn_new)

        context_vectors = torch.matmul(attn_new, v)

        context_vectors = context_vectors.transpose(1, 2).contiguous().view(B, B2, -1)

        return context_vectors

    def forward(self, p_drop = None, gen = None):

        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)

        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent

        # rep_ent_str_new = self.get_multi_attn(rep_ent_str, rep_ent_str, rep_ent_str, 0)
        # #q_vis = rep_ent_str.expand(-1, 8, -1)
        # rep_ent_vis_new = self.get_multi_attn(rep_ent_vis, rep_ent_vis, rep_ent_vis, 1)
        # #q_txt = rep_ent_str.expand(-1, 8, -1)
        # rep_ent_txt_new = self.get_multi_attn(rep_ent_txt, rep_ent_txt, rep_ent_txt, 1)

        # ent_seq_new = torch.cat([rep_ent_str, rep_ent_vis, rep_ent_txt],dim = 1)
        # ent_seq_new = self.get_multi_attn(ent_seq_new, ent_seq_new, ent_seq_new, 1)
        # ent_seq = torch.cat([ent_tkn, ent_seq_new], dim = 1)

        # ent_seq, ent_mask = self.ModalityAwareFusion(rep_ent_str, rep_ent_vis, rep_ent_txt, self.ent_vis_mask, self.ent_txt_mask)
        # ent_seq = torch.cat([ent_tkn, ent_seq], dim = 1)
        # # ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        # ent_seq = self.get_multi_attn(ent_seq, ent_seq, ent_seq, 1)
        # ent_mask = torch.cat([self.false_ents, self.false_ents, ent_mask], dim = 1)



        # if (flag == False):
        #     if torch.rand(1).item() < 0.5:
        #         rep_ent_txt *= 0
        #     else:
        #         rep_ent_vis *= 0

        rep_ent_str, rep_ent_vis, rep_ent_txt, ent_mask = self.ModalityAwareFusion(rep_ent_str, rep_ent_vis, rep_ent_txt, self.ent_vis_mask, self.ent_txt_mask, p_drop)
        ent_tkn2 = ent_tkn.squeeze(1)
        ent_seq1 = torch.cat([ent_tkn, rep_ent_str, ], dim=1)
        ent_seq2 = torch.cat([ent_tkn, rep_ent_vis, ], dim=1)
        ent_seq3 = torch.cat([ent_tkn, rep_ent_txt], dim=1)
        rep_ent_str = self.ent_encoder(ent_seq1)[:, 0]
        rep_ent_vis = self.ent_encoder(ent_seq2)[:, 0]
        rep_ent_txt = self.ent_encoder(ent_seq3)[:, 0]

        if(gen == 1):
            rep_ent_vis = self.generator(rep_ent_vis, 1)
            rep_ent_txt = self.generator(rep_ent_txt, 2)

        ent_seq = torch.stack([ent_tkn2, rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1)  # (1500, 4, 256)
        context_vec = self.context_vec
        att_weights = torch.sum(context_vec * ent_seq * self.scale, dim=-1, keepdim=True)
        att_weights = torch.softmax(att_weights, dim=-1)
        ent_embs = torch.sum(att_weights * ent_seq, dim=1)

        #ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = ent_mask)[:,0]
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))
        return torch.cat([ent_embs, self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1)


    def contrastive_loss(self, emb_ent1):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent

        # 交叉融合
        ent_seq, ent_mask = self.ModalityAwareFusion(rep_ent_str, rep_ent_vis, rep_ent_txt, self.ent_vis_mask,
                                                     self.ent_txt_mask)
        ent_seq = torch.cat([ent_tkn, ent_seq], dim=1)
        # ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_seq = self.get_multi_attn(ent_seq, ent_seq, ent_seq, 1)
        ent_mask = torch.cat([self.false_ents, self.false_ents, ent_mask], dim=1)

        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = ent_mask)[:,0]
        emb_ent2 = torch.cat([ent_embs, self.lp_token], dim = 0)
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]

        contrastive_loss = self.contrastive(emb_ent1[select_ents], emb_ent2[select_ents])
        return contrastive_loss

    def contrastive_loss_finegrained(self, emb_ent1):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent

        # rep_ent_str_new = self.get_multi_attn(rep_ent_str, rep_ent_str, rep_ent_str, 0)
        # #q_vis = rep_ent_str.expand(-1, 8, -1)
        # rep_ent_vis_new = self.get_multi_attn(rep_ent_vis, rep_ent_vis, rep_ent_vis, 1)
        # #q_txt = rep_ent_str.expand(-1, 8, -1)
        # rep_ent_txt_new = self.get_multi_attn(rep_ent_txt, rep_ent_txt, rep_ent_txt, 1)
        #
        # ent_seq_new = torch.cat([rep_ent_str_new, rep_ent_vis_new, rep_ent_txt_new], dim=1)
        # ent_seq_new = self.get_multi_attn(ent_seq_new, ent_seq_new, ent_seq_new, 1)
        # ent_seq = torch.cat([ent_tkn, ent_seq_new], dim=1)

        # ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)

        ent_seq, ent_mask = self.ModalityAwareFusion(rep_ent_str, rep_ent_vis, rep_ent_txt, self.ent_vis_mask,
                                                     self.ent_txt_mask)
        ent_seq = torch.cat([ent_tkn, ent_seq], dim=1)
        # ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_seq = self.get_multi_attn(ent_seq, ent_seq, ent_seq, 1)
        ent_mask = torch.cat([self.false_ents, self.false_ents, ent_mask], dim=1)

        # ent_embs: [ent_num, seq_len, embed_dim]
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = ent_mask)
        # torch.Size([15000, 18, 256]) torch.Size([1, 256])
        # print(ent_embs.shape, self.lp_token.shape)
        emb_ent2 = torch.cat([ent_embs[:,0], self.lp_token], dim = 0)
        ent_emb3 = torch.cat([torch.mean(ent_embs, dim=1), self.lp_token], dim = 0)
        ent_emb4 = torch.cat([torch.mean(ent_embs[:, 2: 2 + self.num_vis, :], dim=1), self.lp_token], dim=0)
        ent_emb5 = torch.cat([torch.mean(ent_embs[:, 2 + self.num_vis: -1, :], dim=1), self.lp_token], dim=0)
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]
        contrastive_loss = 0
        for emb in [emb_ent2, ent_emb3, ent_emb4, ent_emb5]:
            contrastive_loss += self.contrastive(emb_ent1[select_ents], emb[select_ents])
        contrastive_loss /= 4
        return contrastive_loss


    def score(self, emb_ent, emb_rel, triplets):
        # args:
        #   emb_ent: [num_ent, emb_dim]
        #   emb_rel: [num_rel, emb_dim]
        #   triples: [batch_size, 3]
        # return:
        #   scores: [batch_size, num_ent]
        h_seq = emb_ent[triplets[:,0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        r_seq = emb_rel[triplets[:,1] - self.num_ent].unsqueeze(dim = 1) + self.pos_rel
        t_seq = emb_ent[triplets[:,2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)

        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ctx_emb = output_dec[triplets == self.num_ent + self.num_rel]
        # indexs = triplets != self.num_ent + self.num_rel
        # indexs[:, 1] = False
        # ent_emb = output_dec[indexs]
        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ctx_emb, rel_emb)
            score = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
        else:
            # output_dec = self.decoder(dec_seq)
            score = torch.inner(ctx_emb, emb_ent[:-1])
        return score