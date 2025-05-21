import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from custom_kernel import custom_kernel
from task import input_t, output_t
from utils import make_match_reference

TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE = True


def cg_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    contiguous grouped GEMM forward pass for MoE.
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)
        x_scale: Input tensor scales of shape [M_total, 1]
        w_scale: Expert weight tensor scales of shape [num_experts, N]
    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Check if inputs are properly aligned
    M_total, K = inputs.shape
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=torch.bfloat16)

    # Calculate grid size for the kernel
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = (NUM_SMS, 1, 1)
    # Launch kernel
    _kernel_cg_persistent_forward[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=NUM_SMS,
    )

    return output


def create_indices_from_offsets_nosync(m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Create m_indices tensor from m_offsets tensor without CPU-GPU sync points.

    Args:
        m_offsets: Tensor containing cumulative offsets for each group
            e.g., [128, 128, 256, 384, 640, ...]

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    # Get total size from the last offset
    total_size = m_offsets[-1]

    # Pre-allocate output tensor
    indices = torch.empty(total_size, device=m_offsets.device, dtype=torch.int32)

    # Create a range tensor for each section
    prev_offset = torch.zeros(1, device=m_offsets.device, dtype=m_offsets.dtype)

    for i in range(len(m_offsets)):
        # Calculate current section size
        section_size = m_offsets[i] - prev_offset

        # Only fill if section has elements
        if section_size > 0:
            indices[prev_offset : m_offsets[i]] = i

        # Update prev_offset for next iteration
        prev_offset = m_offsets[i]

    return indices


class GroupGEMMStrategy:
    """Base class for group gemm strategies"""

    def __init__(self, custom_activation):
        self.activation_function = custom_activation

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prepare expert weights, including prescaling

        Args:
            all_weights: List of weight tensors from each expert
            submod_name: Name of the submodule (e.g., 'gate_proj', 'up_proj', 'down_proj')
            module: The parent module that will store the arranged weights

        Returns:
            Tensor: The arranged weights in the format required by the specific strategy
        """

        raise NotImplementedError("Requires arrange_expert_weights method")

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the group gemm operation

        Args:
            contig_tokens: The input tokens, arranged contiguously by expert
            m_sizes: Sizes of each group
            m_offsets: Offsets of each group
            module: The MoE module containing weights and parameters

        Returns:
            The processed tokens
        """
        raise NotImplementedError("GroupGEMM strategy must implement execute method")

    @staticmethod
    def is_available() -> bool:
        """Check if this strategy is available on the current system"""
        return False


class TritonCGBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation of Triton Contiguous group Gemm"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""

        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = create_indices_from_offsets_nosync(m_offsets)

        gate_proj = cg_grouped_gemm_forward(valid_tokens, w_gate, m_indices)

        up_proj = cg_grouped_gemm_forward(valid_tokens, w_up, m_indices)

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)

        down_proj_out = cg_grouped_gemm_forward(hidden_outputs, w_down, m_indices)

        # Copy results back to contig_tokens
        contig_tokens[: m_offsets[-1]] = down_proj_out
        return contig_tokens

    @staticmethod
    def is_available() -> bool:
        return TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE


class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(config["n_routed_experts"])]
        )
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(
            x_flat, flat_expert_indices, flat_expert_weights
        )

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    # def forward(self, hidden_states):
    #     identity = hidden_states
    #     orig_shape = hidden_states.shape
    #     # for each token, select top-k experts, and compute the weight for each expert
    #     topk_idx, topk_weight = self.gate(hidden_states)
    #     hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    #     if self.shuffle_method == "symm_mem":
    #         y = self.moe_on_device(hidden_states, topk_idx, topk_weight)
    #     else:  # "torch_all_to_all"
    #         y = self.moe_forward(hidden_states, topk_idx, topk_weight)

    #     y = y.view(*orig_shape)
    #     if self.config.n_shared_experts is not None:
    #         y = y + self.shared_experts(identity)
    #     return y

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )

        return expert_cache

    def moe_forward(self, x, topk_ids, topk_weight):
        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)

        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # all to all
        # This part exchange the information about the number of tokens send and
        # received by each expert. We can understand this information as "side
        # band", which is not part of the actual data. Thus no gradient is
        # needed.

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(
                tokens_per_expert_group, tokens_per_expert, group=self.ep_group
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)

        # DP to EP token shuffle. This part needs gradient.
        if self.shuffle_method == "symm_mem":
            # Move input to the `token_send_buf` symm mem
            token_send_buf = self.get_send_buf()
            token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
            # Note: `out=` avoids copy, but it is not differentiable
            # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=self.token_send_buf[: idxs.shape[0]])
            token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
                token_send_buf,
                input_splits,
                self.ep_group,
            )
            with torch.no_grad():
                # Received tokens from all other ranks. TODO: use mask instead
                received = output_splits.sum()
            # TODO: don't use `received`
            gathered_tokens = token_gather_buf[:received]
        else:  # "torch_all_to_all"
            # Prepare input ans output splits
            with torch.no_grad():
                output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(
                    dim=1
                )
            gathered_tokens = all_to_all_single_autograd(
                sorted_tokens,
                output_splits.tolist(),
                input_splits.tolist(),
                self.ep_group,
            )

        # This part prepares a 1D tensor with the same length as
        # `gathered_tokens`. The 1D tensor is filled with local expert IDs which
        # the tokens in `gathered_tokens` are headed for. This part doesn't need
        # gradient.
        with torch.no_grad():
            gatherd_idxs = (
                torch.arange(
                    tokens_per_expert_group.numel(),
                    device=tokens_per_expert_group.device,
                )
                % self.experts_per_rank
            )
            gatherd_idxs = gatherd_idxs.repeat_interleave(tokens_per_expert_group)

        # Prepare buffer for tokens processed by experts
        if self.shuffle_method == "symm_mem":
            # Take necessary space from `token_gather_buf` symm mem because we are
            # going to send them out after expert processing
            processed_tokens = self.get_gather_buf()[: gathered_tokens.shape[0]]
        else:  # "torch_all_to_all"
            processed_tokens = torch.empty_like(gathered_tokens)

        # This part processes the tokens routed to the local experts.
        # TODO: can we use group GEMM here?
        for i, expert in enumerate(self.experts.values()):
            processed_tokens[gatherd_idxs == i] = expert(
                gathered_tokens[gatherd_idxs == i]
            )

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        if self.shuffle_method == "symm_mem":
            token_return_buf, _ = OnDeviceAllToAllV.apply(
                processed_tokens,
                output_splits,
                self.ep_group,
            )
            returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        else:  # "torch_all_to_all"
            returned_tokens = all_to_all_single_autograd(
                processed_tokens,
                input_splits.tolist(),
                output_splits.tolist(),
                self.ep_group,
            )

        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens
        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )
        return final_out

    def combine_experts(self, submod_name: str):
        all_weights = []
        for expert in self.experts.values():

            lin = expert.get_submodule(submod_name)
            all_weights.append(lin.weight)
            lin.weight = None

        # let the group gemm strategy prep the final weight layout
        combined_weight = self.group_gemm_instance.arrange_expert_weights(
            all_weights, submod_name, self
        )

        if combined_weight is None:
            raise NotImplementedError("expert weights not handled by group gemmm")

        self.register_parameter(f"{submod_name}_weight", nn.Parameter(combined_weight))

    def setup_symm_mem(self, dtype: torch.dtype, device: torch.device):
        # Switch shuffle method
        self.shuffle_method = "symm_mem"

        # Combine expert weights
        self.combine_experts("gate_proj")
        self.combine_experts("up_proj")
        self.combine_experts("down_proj")

        # Assuming worst case, 2x tokens are routed to one EP rank
        overflow = 2
        OnDeviceAllToAllV.max_output_len = (
            self.config.max_seq_len * self.num_experts_per_tok * overflow
        )

        # Symmetric memory buffers are shared by all MoE instances across
        # layers, we only need to initialize them once
        if MoE.token_send_buf is not None:
            return

        # Input buffer for DP-to-EP shuffle
        MoE.token_send_buf = symm_mem.empty(
            self.config.max_seq_len
            * self.num_experts_per_tok,  # seq len * top k (flattened)
            self.config.hidden_size,  # hidden dim
            dtype=dtype,
            device=device,
        )
        # Input buffer for EP-to-DP shuffle
        MoE.token_gather_buf = symm_mem.empty(
            self.config.max_seq_len
            * self.num_experts_per_tok  # seq len * top k (flattened)
            * overflow,
            self.config.hidden_size,  # hidden dim
            dtype=dtype,
            device=device,
        )


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of DeepSeek-style Mixture of Experts using PyTorch.

    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_dim]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters

    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights["router.weight"])

    for i in range(num_experts):
        gate_proj_weight = weights[f"experts.{i}.0.weight"]
        up_proj_weight = weights[f"experts.{i}.1.weight"]
        down_proj_weight = weights[f"experts.{i}.2.weight"]

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t())
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t())
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t())

    moe.shared_expert.W_gate.weight = nn.Parameter(
        weights["shared_experts.0.weight"].t()
    )
    moe.shared_expert.W_up.weight = nn.Parameter(weights["shared_experts.1.weight"].t())
    moe.shared_expert.W_down.weight = nn.Parameter(
        weights["shared_experts.2.weight"].t()
    )

    output = moe(input_tensor)

    return output


def generate_input(
    dhidden: int,
    dexpert: int,
    nroutedexperts: int,
    nsharedexperts: int,
    nexpertspertoken: int,
    bs: int,
    seqlen: int,
    seed: int,
) -> input_t:

    # Really dumb but for now _ isn't parsing correctly.
    d_hidden = dhidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    n_shared_experts = nsharedexperts
    n_experts_per_token = nexpertspertoken
    batch_size = bs
    seq_len = seqlen

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_experts_per_token": n_experts_per_token,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    num_experts = n_routed_experts
    expert_dim = d_expert
    weights = {}

    input_tensor = torch.randn(
        (batch_size, seq_len, d_hidden),
        device="cuda",
        dtype=torch.float16,
        generator=gen,
    ).contiguous()

    # Initialize router weights
    weights["router.weight"] = torch.randn(
        (num_experts, d_hidden), device="cuda", dtype=torch.float16, generator=gen
    ) / math.sqrt(d_hidden)

    for i in range(num_experts):
        weights[f"experts.{i}.0.weight"] = torch.randn(
            (d_hidden, expert_dim), device="cuda", dtype=torch.float16, generator=gen
        ) / math.sqrt(expert_dim)

        weights[f"experts.{i}.1.weight"] = torch.randn(
            (d_hidden, expert_dim), device="cuda", dtype=torch.float16, generator=gen
        ) / math.sqrt(expert_dim)

        weights[f"experts.{i}.2.weight"] = torch.randn(
            (expert_dim, d_hidden), device="cuda", dtype=torch.float16, generator=gen
        ) / math.sqrt(d_hidden)

    weights["shared_experts.0.weight"] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts),
        device="cuda",
        dtype=torch.float16,
        generator=gen,
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights["shared_experts.1.weight"] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts),
        device="cuda",
        dtype=torch.float16,
        generator=gen,
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights["shared_experts.2.weight"] = torch.randn(
        (expert_dim * n_shared_experts, d_hidden),
        device="cuda",
        dtype=torch.float16,
        generator=gen,
    ) / math.sqrt(d_hidden)

    return (input_tensor, weights, config)


if __name__ == "__main__":
    data = generate_input(
        dhidden=512,
        dexpert=128,
        nroutedexperts=16,
        nsharedexperts=1,
        nexpertspertoken=3,
        bs=2,
        seqlen=256,
        seed=42,
    )
    check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
    output = custom_kernel(data)
    print(check_implementation(data, output))
