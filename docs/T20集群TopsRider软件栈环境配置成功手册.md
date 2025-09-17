
-----

### **T20集群TopsRider软件栈环境配置成功手册**

这是我们从零开始，历经所有排错过程，最终总结出的在新机器上部署这套环境的完整、正确、可重复的操作步骤。

#### **背景**

  * **目标硬件**: 燧原T20集群
  * **目标软件**: TopsRider 软件栈 V2.1
  * **操作系统**: Ubuntu 18.04 (主机), Ubuntu 20.04 (容器)
  * **核心挑战**: 服务器公网访问受限，Python依赖环境复杂。

-----

#### **阶段一：主机驱动安装**

此阶段在\*\*T20服务器主机（Host）\*\*上操作。

1.  **获取安装包**：将`TopsRider_t2x_2.5.136_deb_amd64.run`安装包上传到主机。

2.  **确保环境纯净**：为避免旧版本干扰，先执行一遍卸载命令。

    ```bash
    # (此命令在有旧版本时会执行卸载，没有则无操作)
    sudo ./TopsRider_t2x_2.5.136_deb_amd64.run --uninstall
    ```

3.  **安装核心驱动**：执行静默安装，**不带**`--peermem`参数（因为我们确认了缺少Mellanox OFED依赖）。

    ```bash
    sudo ./TopsRider_t2x_2.5.136_deb_amd64.run -y
    ```

4.  **验证驱动加载**：

    ```bash
    lsmod | grep enflame
    ```

    看到有`enflame`相关的输出，即为成功。

-----

#### **阶段二：准备Docker基础环境**

此阶段涉及**本地电脑（例如Mac）和T20服务器主机**。

1.  **[在本地电脑上]** **准备Docker镜像**：
    由于服务器无法直接`docker pull`，现在网络好的本地电脑上准备好`ubuntu:20.04`的镜像。

    ```bash
    # 在本地电脑上执行
    docker pull ubuntu:20.04
    docker save -o ubuntu-20.04.tar ubuntu:20.04
    ```

2.  **[在本地电脑上]** **上传Docker镜像**：
    使用`scp`将镜像文件上传到服务器。

    ```bash
    # 将 [本地路径] 替换为文件的实际路径
    scp -P 60025 [本地路径]/ubuntu-20.04.tar root@117.156.108.234:~/
    ```

3.  **[在服务器主机上]** **加载Docker镜像**：

    ```bash
    docker load -i ~/ubuntu-20.04.tar
    ```

-----

#### **阶段三：容器内软件栈安装与配置**

此阶段在**T20服务器主机**上操作，用于创建和配置容器。

1.  **创建并启动容器**：使用`ubuntu:20.04`镜像，并赋予必要权限。

    ```bash
    docker run -dit --name t20_training_env_py38 --privileged --ipc=host --network=host ubuntu:20.04
    ```

2.  **复制安装包到容器**：

    ```bash
    docker cp TopsRider_t2x_2.5.136_deb_amd64.run t20_training_env_py38:/
    ```

3.  **进入容器**：

    ```bash
    docker exec -it t20_training_env_py38 /bin/bash
    ```

4.  **[在容器内]** **安装基础软件栈和torch-gcu框架**：
    这是最关键的一步，分两步执行，确保`torch-gcu`被正确安装。

    ```bash
    # 容器内执行:
    # 1. 安装基础软件
    ./TopsRider_t2x_2.5.136_deb_amd64.run -y

    # 2. 安装Pytorch GCU框架
    ./TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu
    ```

5.  **[在容器内]** **安装系统依赖库**：
    为后续的Python包和脚本准备所需的系统库。

    ```bash
    apt-get update
    apt-get install -y libelf1 git git-lfs vim
    ```

-----

#### **阶段四：准备并运行验证程序**

此阶段涉及**本地电脑**和**T20服务器容器**。

1.  **[在本地电脑上]** **准备模型和Tokenizer文件**：

    ```bash
    # 在本地电脑上执行:
    # 创建一个目录用于下载
    mkdir ~/sdxl_download
    cd ~/sdxl_download

    # 克隆模型库，这将下载所有文件，包括模型和tokenizer
    git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    ```

2.  **[在本地电脑上]** **上传模型和Tokenizer文件**：
    将模型文件和两个`tokenizer`文件夹上传到**服务器主机**。

    ```bash
    # 在本地电脑上执行:
    # 上传主模型文件
    scp -P 60025 ~/sdxl_download/stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors root@117.156.108.234:/root/

    # 上传tokenizer文件夹 (使用-r递归复制)
    scp -r -P 60025 ~/sdxl_download/stable-diffusion-xl-base-1.0/tokenizer root@117.156.108.234:/root/
    scp -r -P 60025 ~/sdxl_download/stable-diffusion-xl-base-1.0/tokenizer_2 root@117.156.108.234:/root/
    ```

3.  **[在服务器主机上]** **将文件复制到容器内**：

    ```bash
    # 在服务器主机上执行:
    # 定义目标路径变量，简化命令
    TARGET_DIR="/usr/local/topsrider/ai_development_toolkit/huggingface-gcu/sd_scripts_1.1.28/sd_models/stable-diffusion-xl-base-1.0/"

    # 在容器内创建目标目录
    docker exec t20_training_env_py38 mkdir -p ${TARGET_DIR}

    # 将文件从主机复制到容器
    docker cp ~/sd_xl_base_1.0_0.9vae.safetensors t20_training_env_py38:${TARGET_DIR}
    docker cp ~/tokenizer t20_training_env_py38:${TARGET_DIR}
    docker cp ~/tokenizer_2 t20_training_env_py38:${TARGET_DIR}
    ```

4.  **[在容器内]** **安装Python依赖并修复脚本**：

    ```bash
    # 容器内执行:
    # 进入工作目录
    cd /usr/local/topsrider/ai_development_toolkit/huggingface-gcu/sd_scripts_1.1.28/

    # 运行依赖安装脚本
    bash install_for_sd_scripts.sh

    # 手动修复被脚本破坏的依赖环境
    pip3 uninstall -y torch torchvision
    pip3 install "tokenizers>=0.11.1,<0.14" protobuf==3.20.3 huggingface-hub==0.15.1
    pip3 install torchvision==0.11.1 --no-deps
    cd / && ./TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu # 重新安装torch-gcu

    # 进入脚本目录
    cd /usr/local/topsrider/ai_development_toolkit/huggingface-gcu/sd_scripts_1.1.28/

    # 修复Python脚本bug
    # (此处省略手动编辑，直接提供最终的正确脚本)
    ```

    **注意**: 为避免再次手动编辑，理想情况下应将下面提供的最终版 `sdxl_minimal_inference.py` 脚本内容保存为一个文件，并上传替换掉容器内的原始脚本。

5.  **[在容器内]** **最终运行**：
    确保所有文件就位、脚本修复后，执行最终验证。

    ```bash
    bash sdxl_minimal_inference.sh
    ```

    看到`tqdm`进度条开始走，即代表 **“征服T20集群”** 任务圆满成功。

-----

#### **附录：最终修复版的 `sdxl_minimal_inference.py`**

\<details\>
\<summary\>点击展开查看完整脚本\</summary\>

```python
# 手元で推論を行うための最低限のコード。HuggingFace／DiffusersのCLIP、schedulerとVAEを使う
# Minimal code for performing inference at local. Use HuggingFace/Diffusers CLIP, scheduler and VAE

import argparse
import datetime
import math
import os
import random
from einops import repeat
import numpy as np
import torch
import ptex
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler
from PIL import Image
import open_clip
from safetensors.torch import load_file

from library import model_util, sdxl_model_util
import networks.lora as lora

# scheduler: このあたりの設定はSD1/2と同じでいいらしい
# scheduler: The settings around here seem to be the same as SD1/2
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


# Time EmbeddingはDiffusersからのコピー
# Time Embedding is copied from Diffusers


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    # x = rearrange(x, "b d -> (b d)")
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    # emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


if __name__ == "__main__":
    # 画像生成条件を変更する場合はここを変更 / change here to change image generation conditions

    # SDXLの追加のvector embeddingへ渡す値 / Values to pass to additional vector embedding of SDXL
    target_height = 1024
    target_width = 1024
    original_height = target_height
    original_width = target_width
    crop_top = 0
    crop_left = 0

    steps = 50
    guidance_scale = 7
    seed = None  # 1

    DEVICE = ptex.device("xla")
    DTYPE = torch.float16  # bfloat16 may work

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--prompt2", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.prompt2 is None:
        args.prompt2 = args.prompt

    # Use absolute paths to prevent ambiguity
    text_encoder_1_name = "/usr/local/topsrider/ai_development_toolkit/huggingface-gcu/sd_scripts_1.1.28/sd_models/stable-diffusion-xl-base-1.0/tokenizer"
    text_encoder_2_name = "/usr/local/topsrider/ai_development_toolkit/huggingface-gcu/sd_scripts_1.1.28/sd_models/stable-diffusion-xl-base-1.0/tokenizer_2"

    # checkpointを読み込む。モデル変換についてはそちらの関数を参照
    # Load checkpoint. For model conversion, see this function

    # 本体RAMが少ない場合はGPUにロードするといいかも
    # If the main RAM is small, it may be better to load it on the GPU
    text_model1, text_model2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.ckpt_path, "cpu"
    )

    # Text Encoder 1はSDXL本体でもHuggingFaceのものを使っている
    # In SDXL, Text Encoder 1 is also using HuggingFace's

    # Text Encoder 2はSDXL本体ではopen_clipを使っている
    # それを使ってもいいが、SD2のDiffusers版に合わせる形で、HuggingFaceのものを使う
    # 重みの変換コードはSD2とほぼ同じ
    # In SDXL, Text Encoder 2 is using open_clip
    # It's okay to use it, but to match the Diffusers version of SD2, use HuggingFace's
    # The weight conversion code is almost the same as SD2

    # VAEの構造はSDXLもSD1/2と同じだが、重みは異なるようだ。何より謎のscale値が違う
    # fp16でNaNが出やすいようだ
    # The structure of VAE is the same as SD1/2, but the weights seem to be different. Above all, the mysterious scale value is different.
    # NaN seems to be more likely to occur in fp16

    unet.to(DEVICE, dtype=DTYPE)
    unet.eval()

    vae_dtype = DTYPE
    if DTYPE == torch.float16:
        print("use float32 for vae")
        vae_dtype = torch.float32
    vae.to(DEVICE, dtype=vae_dtype)
    vae.eval()

    text_model1.to(DEVICE, dtype=DTYPE)
    text_model1.eval()
    text_model2.to(DEVICE, dtype=DTYPE)
    text_model2.eval()

    unet.set_use_memory_efficient_attention(False, False)
    if torch.__version__ >= "2.0.0": # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(True)

    # Tokenizers
    print("loading tokenizers directly...")
    tokenizer1 = CLIPTokenizer.from_pretrained(text_encoder_1_name, local_files_only=True)
    tokenizer2 = CLIPTokenizer.from_pretrained(text_encoder_2_name, local_files_only=True)
    print("tokenizers loaded.")
    
    # LoRA
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        lora_model, weights_sd = lora.create_network_from_weights(
            multiplier, weights_file, vae, [text_model1, text_model2], unet, None, True
        )
        lora_model.merge_to([text_model1, text_model2], unet, weights_sd, DTYPE, DEVICE)

    # scheduler
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )

    def generate_image(prompt, prompt2, negative_prompt, seed=None):
        # 将来的にサイズ情報も変えられるようにする / Make it possible to change the size information in the future
        # prepare embedding
        with torch.no_grad():
            # vector
            emb1 = get_timestep_embedding(torch.FloatTensor([original_height, original_width]).unsqueeze(0), 256)
            emb2 = get_timestep_embedding(torch.FloatTensor([crop_top, crop_left]).unsqueeze(0), 256)
            emb3 = get_timestep_embedding(torch.FloatTensor([target_height, target_width]).unsqueeze(0), 256)
            # print("emb1", emb1.shape)
            c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(DEVICE, dtype=DTYPE)
            uc_vector = c_vector.clone().to(DEVICE, dtype=DTYPE)  # ちょっとここ正しいかどうかわからない I'm not sure if this is right

            # crossattn

        # Text Encoderを二つ呼ぶ関数  Function to call two Text Encoders
        def call_text_encoder(text, text2):
            global tokenizer1, tokenizer2
            # text encoder 1
            batch_encoding = tokenizer1(
                text,
                truncation=True,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(DEVICE)

            with torch.no_grad():
                enc_out = text_model1(tokens, output_hidden_states=True, return_dict=True)
                text_embedding1 = enc_out["hidden_states"][11]
                # text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)   # layer normは通さないらしい

            # text encoder 2
            batch_encoding_2 = tokenizer2(
                text2,
                truncation=True,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            tokens = batch_encoding_2["input_ids"].to(DEVICE)

            with torch.no_grad():
                enc_out = text_model2(tokens, output_hidden_states=True, return_dict=True)
                text_embedding2_penu = enc_out["hidden_states"][-2]
                # print("hidden_states2", text_embedding2_penu.shape)
                text_embedding2_pool = enc_out["text_embeds"]   # do not support Textual Inversion

            # 連結して終了 concat and finish
            text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2)
            return text_embedding, text_embedding2_pool

        # cond
        c_ctx, c_ctx_pool = call_text_encoder(prompt, prompt2)
        # print(c_ctx.shape, c_ctx_p.shape, c_vector.shape)
        c_vector = torch.cat([c_ctx_pool, c_vector], dim=1)

        # uncond
        uc_ctx, uc_ctx_pool = call_text_encoder(negative_prompt, negative_prompt)
        uc_vector = torch.cat([uc_ctx_pool, uc_vector], dim=1)

        text_embeddings = torch.cat([uc_ctx, c_ctx])
        vector_embeddings = torch.cat([uc_vector, c_vector])

        # メモリ使用量を減らすにはここでText Encoderを削除するかCPUへ移動する

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            ptex.tops.manual_seed_all(seed)

            # # random generator for initial noise
            # generator = torch.Generator(ptex.device("xla")).manual_seed(seed)
            generator = None
        else:
            generator = None

        # get the initial random noise unless the user supplied it
        # SDXLはCPUでlatentsを作成しているので一応合わせておく、Diffusersはtarget deviceでlatentsを作成している
        # SDXL creates latents in CPU, Diffusers creates latents in target device
        latents_shape = (1, 4, target_height // 8, target_width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(DEVICE, dtype=DTYPE)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

        # set timesteps
        scheduler.set_timesteps(steps, DEVICE)

        # このへんはDiffusersからのコピペ
        # Copy from Diffusers
        timesteps = scheduler.timesteps.to(DEVICE)  # .to(DTYPE)
        num_latent_input = 2
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                noise_pred = unet(latent_model_input, t, text_embeddings, vector_embeddings)

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # latents = 1 / 0.18215 * latents
            latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents
            latents = latents.to(vae_dtype)
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # image = self.numpy_to_pil(image)
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        # 保存して終了 save and finish
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for i, img in enumerate(image):
            img.save(os.path.join(args.output_dir, f"image_{timestamp}_{i:03d}.png"))

    if not args.interactive:
        generate_image(args.prompt, args.prompt2, args.negative_prompt, seed)
    else:
        # loop for interactive
        while True:
            prompt = input("prompt: ")
            if prompt == "":
                break
            prompt2 = input("prompt2: ")
            if prompt2 == "":
                prompt2 = prompt
            negative_prompt = input("negative prompt: ")
            seed = input("seed: ")
            if seed == "":
                seed = None
            else:
                seed = int(seed)
            generate_image(prompt, prompt2, negative_prompt, seed)

    print("Done!")
```

\</details\>