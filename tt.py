import torch
import torch.nn as nn
import time

class Attention(nn.Module):
    """
    用户提供的原始Attention模块。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # --- FIX 1: Store dim and head_dim as instance attributes ---
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # ------------------------------------------------------------
        
        self.scale = self.head_dim ** -0.5

        # QKV线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
     
        # 用于masking的门，1代表保留，0代表剪枝
        # 在实际剪枝场景中，这个gate会通过训练得到
        self.gate = torch.ones(self.num_heads)

    def forward(self, x):
        B, N, C_in = x.shape
        # --- FIX 2: Use self.head_dim for reshaping qkv ---
        # 这确保了即使输入维度C_in与模块内部维度self.dim不同，也能正确计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # ----------------------------------------------------
        q, k, v = qkv.unbind(0)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        x = (attn @ v).transpose(1, 2) # -> [B, N, num_heads, head_dim]

        # 应用gate进行masking
        # 这是剪枝/稀疏化的关键步骤
        mask = self.gate.to(x.device)
        x = x * mask.view(1, 1, self.num_heads, 1)

        # --- FIX 3: Use self.dim for the final reshape ---
        # 将输出重塑为模块自身的维度，而不是输入的维度
        x = x.reshape(B, N, self.dim)
        # -------------------------------------------------
        
        # 为了方便验证，我们只返回x
        return x

def compress_attention(attention_module: Attention, head_mask: torch.Tensor) -> Attention:
    """
    将一个带有mask的Attention模块压缩成一个更小的、等效的密集模块。

    Args:
        attention_module (Attention): 原始的、待压缩的Attention模块。
        head_mask (torch.Tensor): 一个形状为 (num_heads,) 的张量, 
                                  用1表示保留的头，用0表示要剪掉的头。

    Returns:
        Attention: 一个新的、更小的、计算上等效的Attention模块实例。
    """
    # 1. 获取原始模块的参数
    orig_qkv = attention_module.qkv
    old_dim = orig_qkv.in_features
    old_num_heads = attention_module.num_heads
    head_dim = old_dim // old_num_heads
    has_bias = orig_qkv.bias is not None

    # 2. 识别需要保留的头
    kept_indices = torch.where(head_mask != 0)[0].cpu()
    new_num_heads = len(kept_indices)

    if new_num_heads == 0:
        raise ValueError("Cannot prune all heads.")
    if new_num_heads == old_num_heads:
        print("No heads to prune. Returning a deep copy of the original module.")
        # deepcopy is not a standard method, using a manual copy
        import copy
        return copy.deepcopy(attention_module)


    new_dim = new_num_heads * head_dim

    # 3. 创建一个新的 "模板" Attention 对象
    # 我们用 new_dim 和 new_num_heads 初始化它。
    # 这可以确保新模块内部的 head_dim, scale, 和最终的 reshape 逻辑是正确的。
    new_attn = Attention(
        dim=new_dim,
        num_heads=new_num_heads,
        qkv_bias=has_bias,
        attn_drop=attention_module.attn_drop.p,
    )

    # 4. 从原始QKV层提取、切片并加载权重
    W_qkv = orig_qkv.weight.data
    
    # 原始qkv.weight的形状是 (old_dim * 3, old_dim)
    # 它由 W_q, W_k, W_v 垂直堆叠而成，每个都是 (old_dim, old_dim)
    # 在每个 W_q/k/v 内部，权重又是按头（head）连续排列的
    
    # 计算需要保留的权重行的索引
    indices_to_keep_in_qkv_rows = []
    for i in range(3):  # 对应 Q, K, V
        # 每个部分的起始行索引
        part_offset = i * old_dim
        for h_idx in kept_indices:
            # 对应头的起始行索引
            head_start_row = part_offset + h_idx * head_dim
            indices_to_keep_in_qkv_rows.extend(range(head_start_row, head_start_row + head_dim))
    
    indices_to_keep_in_qkv_rows = torch.tensor(indices_to_keep_in_qkv_rows, dtype=torch.long)

    # 新的QKV权重，只保留需要的行，列（输入维度）保持不变
    new_W_qkv = W_qkv[indices_to_keep_in_qkv_rows, :]

    # 5. 创建一个新的、尺寸正确的QKV层并替换模板中的旧层
    # 新的qkv层需要将 old_dim 映射到 new_dim * 3
    correct_qkv = nn.Linear(old_dim, new_dim * 3, bias=has_bias)
    correct_qkv.weight.data.copy_(new_W_qkv)

    # 如果存在偏置，同样处理
    if has_bias:
        B_qkv = orig_qkv.bias.data
        new_B_qkv = B_qkv[indices_to_keep_in_qkv_rows]
        correct_qkv.bias.data.copy_(new_B_qkv)

    # 用我们精心创建的、正确的qkv层替换掉模板中的qkv层
    new_attn.qkv = correct_qkv
    
    # 压缩后的模块不再需要gate，我们将其设置为全1
    new_attn.gate = torch.ones(new_attn.num_heads)

    print(f"Successfully compressed Attention module:")
    print(f"  - Heads: {old_num_heads} -> {new_num_heads}")
    print(f"  - Dimension: {old_dim} -> {new_dim}")
    print(f"  - QKV Layer: {orig_qkv.weight.shape} -> {new_attn.qkv.weight.shape}")

    return new_attn


# --- 验证 ---
if __name__ == '__main__':
    # 定义模型参数
    batch_size = 4
    seq_len = 10
    dim = 768
    num_heads = 12
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 创建原始的Attention模块
    original_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=True).to(device)
    original_attn.eval() # 设置为评估模式

    # 2. 定义一个head mask，例如，我们保留一半的头
    # 保留偶数索引的头: 0, 2, 4, 6, 8, 10
    head_mask = torch.zeros(num_heads)
    head_mask[::4] = 1.0
    
    # 将mask应用到原始模块的gate上
    original_attn.gate = head_mask
    print(f"Original module's gate set to: {original_attn.gate.tolist()}")

    # 创建一个随机输入
    dummy_input = torch.randn(batch_size, seq_len, dim).to(device)

    # 3. 通过原始的、被mask的模块计算输出
    with torch.no_grad():
        start_time = time.time()
        output_original_masked = original_attn(dummy_input)
        end_time = time.time()
        print(f"\nOriginal (masked) module forward pass took: {(end_time - start_time)*1000:.4f} ms")


    # 4. 压缩模块
    compressed_attn = compress_attention(original_attn, head_mask).to(device)
    print(compressed_attn)
    compressed_attn.eval()

    # 5. 通过压缩后的模块计算输出
    with torch.no_grad():
        start_time = time.time()
        output_compressed = compressed_attn(dummy_input)
        end_time = time.time()
        print(f"Compressed module forward pass took: {(end_time - start_time)*1000:.4f} ms")

    # 6. 验证结果
    print("\n--- Verification ---")
    print(f"Shape of original (masked) output: {output_original_masked.shape}")
    print(f"Shape of compressed output:        {output_compressed.shape}")
    
    # 关键点：压缩后的输出维度更小，因为它只包含有效信息。
    # 为了验证数值是否一致，我们需要从原始输出中提取出对应于保留头的部分。
    
    # 将原始输出重新解释为 per-head 的形式
    # [B, N, D] -> [B, N, num_heads, head_dim]
    output_original_per_head = output_original_masked.reshape(batch_size, seq_len, original_attn.num_heads, original_attn.head_dim)
    
    # 提取保留头对应的部分
    kept_indices = torch.where(head_mask != 0)[0]
    extracted_part = output_original_per_head[:, :, kept_indices, :]
    
    # 将提取出的部分重塑，使其形状与压缩模块的输出一致
    extracted_part_reshaped = extracted_part.reshape(batch_size, seq_len, -1)
    
    print(f"Shape of extracted part from original: {extracted_part_reshaped.shape}")
    
    # 比较两者是否相等
    are_equal = torch.allclose(output_compressed, extracted_part_reshaped, atol=1e-6)
    
    print(f"\nAre the compressed output and the extracted part of the original output numerically equivalent? -> {are_equal}")
    
    if are_equal:
        print("✅ Verification successful! The compressed module correctly reproduces the output of the unpruned heads.")
    else:
        print("❌ Verification failed.")

