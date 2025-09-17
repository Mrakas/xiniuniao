import torch
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
import time
import numpy as np
from tqdm import tqdm
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ======================================================================================
# 1. 配置参数
# ======================================================================================
MODEL_ID = '/apdcephfs_zwfy/share_304071302/arthurzhong/checkpoints/ft_local/FLUX.1-dev'
PROMPT = "A majestic cat knight in shining armor, holding a glowing sword, standing on a castle parapet with a dragon flying in the background under a starry night."

# 树搜索配置
BRANCH_FACTOR = 4     
NUM_INFERENCE_STEPS = 20
NOISE_LEVEL = 0.7
MAX_BATCH_SIZE = 16    

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE =  torch.float16

print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# ======================================================================================
# 2. 加载模型
# ======================================================================================
print("\nLoading FLUX model...")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
pipe.to(DEVICE)
transformer = pipe.transformer
vae = pipe.vae
scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
print("Model loaded successfully.")

# ======================================================================================
# 3. 准备推理所需的输入 
# ======================================================================================
print("\nPreparing inputs for inference...")
# a. 编码 Prompt，获取 embeds 和 text_ids
text_inputs = pipe.encode_prompt(prompt=PROMPT, prompt_2=PROMPT,device=DEVICE)
prompt_embeds = text_inputs[0]
pooled_prompt_embeds = text_inputs[1]
text_ids = text_inputs[2].to(DEVICE)

# b. 使用 pipe.prepare_latents 创建正确的 latents 和 image_ids
height = pipe.default_sample_size * pipe.vae_scale_factor
width = pipe.default_sample_size * pipe.vae_scale_factor
num_channels_latents = transformer.config.in_channels // 4 # 64 // 4 = 16
generator = torch.Generator(device=DEVICE).manual_seed(42)
initial_latents, latent_image_ids = pipe.prepare_latents(
    1, # batch_size
    num_channels_latents,
    height,
    width,
    prompt_embeds.dtype,
    DEVICE,
    generator=generator,
)
print(f"Correct latent shape: {initial_latents.shape}") # 应为 [1, 4096, 64]
print(f"Correct image_ids shape: {latent_image_ids.shape}")

# b. 计算 mu 并设置 timesteps (!!! 基于 3D Latent !!!)
# image_seq_len 现在是 shape[1]
image_seq_len = initial_latents.shape[1]
# c. 计算 mu 并设置 timesteps
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b

# image_seq_len 是 latent 的序列长度，不是像素的
mu = calculate_shift(image_seq_len)
scheduler.set_timesteps(NUM_INFERENCE_STEPS, mu=mu, device=DEVICE)
timesteps = scheduler.timesteps
print("Inputs prepared.")
# ======================================================================================
# 4. 带分支和微批处理的树搜索推理循环
# ======================================================================================
GUIDANCE_SCALE = 7.0
all_leaves_cpu = [initial_latents.cpu()]

with torch.no_grad():
    sigma_max = scheduler.sigmas[1].item()

    for i, t in enumerate(tqdm(timesteps[:-1], desc="SDE Tree Search")):
        num_current_leaves = len(all_leaves_cpu)
        next_gen_leaves_cpu = []
        for batch_idx in range(0, num_current_leaves, MAX_BATCH_SIZE):
            latents_batch_cpu = all_leaves_cpu[batch_idx : batch_idx + MAX_BATCH_SIZE]
            # 确保 latents 是 float32 类型以避免计算溢出，与参考代码保持一致
            latents_gpu = torch.cat(latents_batch_cpu, dim=0).to(DEVICE, dtype=torch.float32)
            current_batch_size = latents_gpu.shape[0]

            guidance = torch.full((current_batch_size,), GUIDANCE_SCALE, device=DEVICE, dtype=DTYPE)
            # d. Transformer 调用 
            model_output = transformer(
                hidden_states=latents_gpu.to(DTYPE), # 将输入转为模型期望的DTYPE
                timestep=(t.expand(current_batch_size) / 1000).to(DTYPE),
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds.expand(current_batch_size, -1),
                encoder_hidden_states=prompt_embeds.expand(current_batch_size, -1, -1),
                txt_ids=text_ids,
                img_ids=latent_image_ids.expand(current_batch_size, -1, -1),
            ).sample.to(torch.float32) # 将输出转回float32进行计算

            # --- SDE 步进逻辑 (优化为标量计算) ---

            # 1. 获取 sigmas 和 dt (作为标量)
            sigma_t_scalar = scheduler.sigmas[i].item() # .item() 获取纯python数字
            sigma_s_scalar = scheduler.sigmas[i + 1].item()
            dt_scalar = sigma_s_scalar - sigma_t_scalar

            # 2. 计算 std_dev_t (作为标量)
            sigma_div_scalar = sigma_max if sigma_t_scalar == 1 else sigma_t_scalar
            # 注意：为避免除零错误，需要对 (1 - sigma_div_scalar) 做保护
            if 1 - sigma_div_scalar == 0:
                # 根据SDE的具体形式决定如何处理，这里假设给一个很大的值或一个小的非零值
                # 或者如果 sigma 永远不会等于1，则可以忽略
                std_dev_t_scalar = float('inf') # 或其他合理的值
            else:
                std_dev_t_scalar = math.sqrt(sigma_t_scalar / (1 - sigma_div_scalar)) * NOISE_LEVEL

            # 3. 计算 prev_sample_mean (父节点均值)
            # PyTorch 会自动将标量广播到张量上
            velocity = model_output
            term1_factor = 1 + std_dev_t_scalar**2 / (2 * sigma_t_scalar) * dt_scalar
            term2_factor = (1 + std_dev_t_scalar**2 * (1 - sigma_t_scalar) / (2 * sigma_t_scalar)) * dt_scalar
            prev_sample_mean_gpu = latents_gpu * term1_factor + velocity * term2_factor

            # 4. 计算用于分支的噪声缩放因子 (作为标量)
            # torch.sqrt(-dt) 在dt为负时会产生NaN，dt = sigma_s - sigma_t, sigma随时间步减小，所以dt是负数，-dt是正数
            noise_scaling_factor = std_dev_t_scalar * math.sqrt(-dt_scalar)


            # 分支逻辑
            for parent_mean_gpu in prev_sample_mean_gpu:
                parent_mean_gpu = parent_mean_gpu.unsqueeze(0)
                for _ in range(BRANCH_FACTOR):
                    # randn_tensor 应该生成与 parent_mean_gpu 形状相同的标准正态分布噪声
                    noise = randn_tensor(parent_mean_gpu.shape, generator=None, device=DEVICE, dtype=torch.float32)
                    child_latent_gpu = parent_mean_gpu + noise * noise_scaling_factor
                    # 将生成的子节点转为CPU存储，并确保数据类型与 all_leaves_cpu 中的张量一致
                    next_gen_leaves_cpu.append(child_latent_gpu.cpu().to(DTYPE)) # 假设 all_leaves_cpu 存储 DTYPE
        
        all_leaves_cpu = next_gen_leaves_cpu

# ======================================================================================
# 5. 解码和保存结果
# ======================================================================================
print(f"\nDecoding all {len(all_leaves_cpu)} samples...")

# 直接使用 all_leaves_cpu 列表进行解码，不再随机抽样
final_latents_to_decode_cpu = all_leaves_cpu

# 在解码前，手动清理一下 GPU 缓存，释放树搜索过程中可能产生的碎片化内存
torch.cuda.empty_cache()

# 准备一个列表来存放解码后的 PIL 图像
decoded_images = []

print("Decoding latents one by one to save memory...")
with torch.no_grad():
    # 迭代所有待解码的 latents
    for i, latent_cpu in enumerate(tqdm(final_latents_to_decode_cpu, desc="Decoding All Images")):
        # 1. 将单个 latent 移动到 GPU
        latent_gpu = latent_cpu.to(DEVICE, dtype=DTYPE) # Shape: [1, 4096, 64]

        # 2. 使用 _unpack_latents 进行重塑
        latent_for_vae = FluxPipeline._unpack_latents(
            latents=latent_gpu,
            height=height,
            width=width,
            vae_scale_factor=pipe.vae_scale_factor
        ) # Shape: [1, 16, 128, 128]

        # 应用 VAE 的缩放和移位因子
        latent_for_vae = (latent_for_vae / vae.config.scaling_factor) + vae.config.shift_factor

        # 解码 latent 得到图像张量
        image_tensor = vae.decode(latent_for_vae, return_dict=False)[0] # Shape: [1, 3, 1024, 1024]
        
        # 后处理为 PIL 图像
        pil_image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
        decoded_images.append(pil_image)

        # 如果 GPU 内存非常紧张
        # torch.cuda.empty_cache()

# 保存所有解码后的图像
print(f"\nSaving all {len(decoded_images)} decoded images...")
for i, img in enumerate(decoded_images):
    # 使用索引 i 命名文件，确保每个文件都有唯一的名字
    save_path = f"/apdcephfs_zwfy/share_304071302/marcuskwan/reward_model/output/flux_2/result_{i}.png"
    
    # 创建保存目录（如果不存在）
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    img.save(save_path)
    print(f"Saved image to: {save_path}")

print(f"\nSuccessfully decoded and saved all {len(decoded_images)} images.")
print("Tree search validation script finished successfully.")

