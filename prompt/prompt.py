import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .utils.layers import GraphConvolution, DistanceAdj
# from clip import clip
from model import longclip as clip
from model.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class CosineCriterion(nn.Module):
    def __init__(self):
        super(CosineCriterion, self).__init__()
        self.eps = 1e-12

    def forward(self, pred, target):
        pred_denom = torch.norm(pred, p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(pred)
        pred = pred / pred_denom
        target_denom = torch.norm(target, p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(target)
        target = target / target_denom

        ret = pred * target
        ret = 1.0 - ret.sum(dim=-1)
        ret = ret.mean()
        return ret

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt
   
class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames):
        super().__init__()
        self.clip_model = clip_model
        n_cls = len(classnames) 
        n_ctx = 16 
        ctx_init = None 
        self.dtype = self.clip_model.dtype
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        CSC = True

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            self.prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                # print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=self.dtype)
            else:
                # print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors) 
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"        
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx.cuda()
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, self.tokenized_prompts

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class PromptCLIP(torch.nn.Module):
    def __init__(self, prompt_size, clip_model, classnames,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 ad_t = False, ad_v = False, coop = False,device = "cuda"
                 ):
        super().__init__()

        delta = torch.zeros((3, prompt_size, prompt_size))
        delta.require_grad = True
        self.perturbation = torch.nn.Parameter(
            delta.float(), requires_grad=True)# 视频提示

        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(self.clip_model, classnames)
        self.classnames = classnames
        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = self.clip_model.encode_text
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.fc = nn.Linear(512, len(classnames)).cuda()
        self.adapter = Adapter(512, 4).to(clip_model.dtype).cuda()
        self.ratio_raw = nn.Parameter(torch.logit(torch.tensor(0.2), eps=1e-6))
        self.ad_t = ad_t
        self.ad_v = ad_v
        self.coop = coop
        # self.fc = nn.Linear(512, len(classnames)).to(self.dtype).to(self.device())

        # self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        # self.embed_dim = embed_dim
        self.attn_window = attn_window
        # self.prompt_prefix = prompt_prefix
        # self.prompt_postfix = prompt_postfix
        self.device = device
        # 只用一层，再增加
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            # attn_mask=self.build_attention_mask(self.attn_window)
            attn_mask=None
        ).cuda()
        # 不用gcn
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True).cuda()
        self.gc2 = GraphConvolution(width, width, residual=True).cuda()
        self.gc3 = GraphConvolution(visual_width, width, residual=True).cuda()
        self.gc4 = GraphConvolution(width, width, residual=True).cuda()
        self.disAdj = DistanceAdj().cuda()
        self.linear = nn.Linear(visual_width, visual_width).cuda()
        self.gelu = QuickGELU().cuda()

        # self.mlp1 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(visual_width, visual_width * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(visual_width * 4, visual_width))
        # ])).cuda()
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ])).cuda()
        self.classifier = nn.Linear(visual_width, 1).cuda()

        # self.clipmodel, _ = clip.load("ViT-B/16", device)
        # for clip_param in self.clipmodel.parameters():
        #     clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width).cuda()
        # self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, b=None,t=None):
        frames_embedding = self.image_encoder(images)

        frames_embedding = frames_embedding.view(b, t, -1)
        images = frames_embedding.to(torch.float)
        # print(images.shape)
        # position_ids = torch.arange(self.visual_length, device=self.device)
        # position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        # frame_position_embeddings = self.frame_position_embeddings(position_ids)
        # frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        # images = images.permute(1, 0, 2) + frame_position_embeddings

        images = images.permute(1, 0, 2)

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        # adj = self.adj4(x, None)
        # disadj = self.disAdj(x.shape[0], x.shape[1])
        # x1_h = self.gelu(self.gc1(x, adj))
        # x2_h = self.gelu(self.gc3(x, disadj))
        #
        # x1 = self.gelu(self.gc2(x1_h, adj))
        # x2 = self.gelu(self.gc4(x2_h, disadj))
        #
        # x = torch.cat((x1, x2), 2)
        # x = self.linear(x)

        return x

    def forward(self, images,b=None,t=None):
        if (self.ad_v == True):
            frames_embedding = self.encode_video(images, b, t)# 时序adapter
            # frames_embedding = frames_embedding + self.mlp2(frames_embedding)
        else:
            frames_embedding = self.image_encoder(images)
            frames_embedding = frames_embedding.view(b,t,-1)
        visual_embedding = torch.mean(frames_embedding, dim=1)

        if (self.ad_t == True):
            text_features = self.text_encoder(clip.tokenize(self.classnames).cuda())
            x = self.adapter(text_features)
            ratio = torch.sigmoid(self.ratio_raw)
            # print(ratio)
            text_features = ratio * x + (1 - ratio) * text_features
        elif (self.coop == True):
            prompts, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, coop=True)
        else:
            text_features = self.text_encoder(clip.tokenize(self.classnames).cuda())


        image_features = visual_embedding / visual_embedding.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        logits = logit_scale * image_features.half() @ text_features.half().t()

        return logits, visual_embedding

    def get_ratio(self):
        return torch.sigmoid(self.ratio_raw)
# class PromptCLIP(torch.nn.Module):
#     def __init__(self, prompt_size, clip_model, classnames):
#         super().__init__()
#         self.clip_model = clip_model
#         self.prompt_learner = PromptLearner(self.clip_model, classnames)
#
#         delta = torch.zeros((3, prompt_size, prompt_size))
#         self.perturbation = torch.nn.Parameter(delta.float(), requires_grad=True)
#
#         self.image_encoder = self.get_attr("encode_image")
#         self.text_encoder = self.get_attr("encode_text")
#         self.logit_scale = self.get_attr("logit_scale")
#         self.dtype = self.get_attr("dtype")
#         self.fc = nn.Linear(512, len(classnames))  # 不加 .cuda()，由模型整体 to(device)
#
#     def get_attr(self, attr_name):
#         return getattr(self.clip_model.module if hasattr(self.clip_model, "module") else self.clip_model, attr_name)
#
#     def forward(self, images, b=None, t=None):
#         device = images.device
#         frames_embedding = self.image_encoder(images)  # assumes shape: (B*T, C)
#         frames_embedding = frames_embedding.view(b, t, -1)
#         visual_embedding = torch.mean(frames_embedding, dim=1)
#
#         prompts, tokenized_prompts = self.prompt_learner()
#         text_features = self.text_encoder(prompts, tokenized_prompts, coop=True)
#
#         image_features = visual_embedding / visual_embedding.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#
#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()
#
#         return logits, visual_embedding
