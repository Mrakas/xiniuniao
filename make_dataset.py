import os
import json
import random
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import argparse
import tqdm

to_tensor = transforms.ToTensor()

def worker_fn(rank, args, all_pairs, tasks_for_this_worker):
    # 1. 设置当前进程应该使用的GPU
    gpu_id = rank % args.n_gpus
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    print(f"[Worker {rank}] Started. Assigned to GPU {gpu_id}.")

    # 2. 在当前进程和指定GPU上加载模型（每个进程仅加载一次）
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        subfolder="vae", 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        subfolder="scheduler"
    )
    num_train_timesteps = scheduler.config.num_train_timesteps

    # 3. 分配任务
    pbar = tqdm.tqdm(tasks_for_this_worker, position=rank, desc=f"Worker {rank} on GPU {gpu_id}")
    for index, timestep in pbar:
        try:
            pair = all_pairs[index]
            image1 = Image.open(os.path.join(args.dataset_path, pair["path1"])).convert("RGB")
            image2 = Image.open(os.path.join(args.dataset_path, pair["path2"])).convert("RGB")
            image_pair_tensor = torch.stack([to_tensor(image1), to_tensor(image2)]) * 2 - 1
            image_pair_tensor = image_pair_tensor.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                latent_pair = vae.encode(image_pair_tensor).latent_dist.sample() * vae.config.scaling_factor
            
            noise = torch.randn_like(latent_pair)
            sigma = (timestep + 1) / num_train_timesteps
            noised_latent_pair = sigma * noise + (1 - sigma) * latent_pair

            output_filename = f"pair_{index}_timestep_{timestep}.pt"
            output_path = os.path.join(args.output_path, "noised_latent_pairs", output_filename)
            torch.save(noised_latent_pair.cpu(), output_path)

        except Exception as e:
            print(f"[Worker {rank}] Error processing task (index={index}, timestep={timestep}): {e}")

    print(f"[Worker {rank}] Finished all its tasks.")


def main(args):
    all_pairs_path = os.path.join(args.dataset_path, "all.json")
    with open(all_pairs_path, "r") as f:
        all_pairs = json.load(f)
    
    indices = range(len(all_pairs))
    output_dir = os.path.join(args.output_path, "noised_latent_pairs")
    os.makedirs(output_dir, exist_ok=True)
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="scheduler")
    all_tasks = []
    for timestep in scheduler.timesteps.long():
        # 对每个 timestep 采样 k 个不同的pair
        sampled_indices = random.sample(indices, k=args.samples_per_timestep)
        for index in sampled_indices:
            all_tasks.append((index, timestep))
    random.shuffle(all_tasks)
    print(f"Total tasks to process: {len(all_tasks)}")

    world_size = args.n_gpus * args.m_tasks
    processes = []
    
    tasks_per_worker = len(all_tasks) // world_size
    
    print(f"Starting {world_size} worker processes...")
    for rank in range(world_size):
        start_idx = rank * tasks_per_worker
        end_idx = (rank + 1) * tasks_per_worker if rank != world_size - 1 else len(all_tasks)
        tasks_for_this_worker = all_tasks[start_idx:end_idx]
        
        p = mp.Process(target=worker_fn, args=(rank, args, all_pairs, tasks_for_this_worker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("All tasks have been completed.")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Parallel data processing for diffusion models.")
    parser.add_argument("--n_gpus", type=int, default=2, help="N: The number of GPUs to use.")
    parser.add_argument("--m_tasks", type=int, default=1, help="M: The number of parallel tasks per GPU.")
    parser.add_argument("--dataset_path", type=str, default="./", help="Path to the dataset containing all.json.")
    parser.add_argument("--output_path", type=str, default="./", help="Path to save the output files.")
    parser.add_argument("--samples_per_timestep", type=int, default=1, help="Number of samples to generate for each timestep.")
    
    args = parser.parse_args()
    
    main(args)