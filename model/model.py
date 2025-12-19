import math

import torch
from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
	model_type = 'minimind'

	def __init__(
		self,
		dropout: float = 0.0,
		bos_token_id: int = 1,
		eos_token_id: int = 2,
		hidden_act: str = 'silu',
		hidden_size: int = 512,
		intermediate_size: int | None = None,
		max_position_embeddings: int = 32768,
		num_attention_heads: int = 8,
		num_hidden_layers: int = 8,
		num_key_value_heads: int = 2,
		vocab_size: int = 6400,
		rms_norm_eps: float = 1e-05,
		rope_theta: float = 1000000.0,
		inference_rope_scaling: bool = False,
		flash_attn: bool = True,
		####################################################
		# Here are the specific configurations of MOE
		# When use_moe is false, the following is invalid
		####################################################
		use_moe: bool = False,
		num_experts_per_tok: int = 2,
		n_routed_experts: int = 4,
		n_shared_experts: int = 1,
		scoring_func: str = 'softmax',
		aux_loss_alpha: float = 0.01,
		seq_aux: bool = True,
		norm_topk_prob: bool = True,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.dropout = dropout
		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id
		self.hidden_act = hidden_act
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.max_position_embeddings = max_position_embeddings
		self.num_attention_heads = num_attention_heads
		self.num_hidden_layers = num_hidden_layers
		self.num_key_value_heads = num_key_value_heads
		self.vocab_size = vocab_size
		self.rms_norm_eps = rms_norm_eps
		self.rope_theta = rope_theta
		self.inference_rope_scaling = inference_rope_scaling
		# 外推长度 = factor * original_max_position_embeddings = 32768
		self.rope_scaling = (
			{
				'beta_fast': 32,
				'beta_slow': 1,
				'factor': 16,
				'original_max_position_embeddings': 2048,
				'attention_factor': 1.0,
				'type': 'yarn',
			}
			if self.inference_rope_scaling
			else None
		)
		self.flash_attn = flash_attn
		####################################################
		# Here are the specific configurations of MOE
		# When use_moe is false, the following is invalid
		####################################################
		self.use_moe = use_moe
		self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
		self.n_routed_experts = n_routed_experts  # 总的专家数量
		self.n_shared_experts = n_shared_experts  # 共享专家
		self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
		self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
		self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
		self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class RMSNorm(torch.nn.Module):
	def __init__(self, dim: int, eps: float = 1e-5):
		super().__init__()
		self.eps = eps
		self.dim = dim
		self.weight = torch.nn.Parameter(torch.ones(dim))

	def _norm(self, x: torch.Tensor):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x: torch.Tensor):
		return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(
	dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict | None = None
):
	freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	attn_factor = 1.0
	if rope_scaling is not None:
		orig_max, factor, beta_fast, beta_slow, attn_factor = (
			rope_scaling.get('original_max_position_embeddings', 2048),
			rope_scaling.get('factor', 16),
			rope_scaling.get('beta_fast', 32),
			rope_scaling.get('beta_slow', 1),
			rope_scaling.get('attention_factor', 1.0),
		)
		if end / orig_max > 1.0:

			def inv_dim(b):
				return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

			low = max(math.floor(inv_dim(beta_fast)), 0)
			high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
			ramp = torch.clamp(
				(torch.arange(dim // 2, device=freqs.device).float() - low)
				/ max(high - low, 0.001),
				0,
				1,
			)
			freqs = freqs * (1 - ramp + ramp / factor)

	t = torch.arange(end, device=freqs.device)
	freqs = torch.outer(t, freqs).float()
	freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
	freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
	return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
	def rotate_half(x):
		return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

	q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
	k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
	return q_embed, k_embed
