"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from utils_cubemap import rotate_camera
from utils_mitsuba import cubemap_to_panorama_torch


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        means[:, 0] *= -1
        means[:, -1] *= -1

        # # crop
        # aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
        # edges = aabb[3:] - aabb[:3]
        # sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
        # sel = torch.where(sel)[0]
        # means, quats, scales, colors, opacities = (
        #     means[sel],
        #     quats[sel],
        #     scales[sel],
        #     colors[sel],
        #     opacities[sel],
        # )

        # # repeat the scene into a grid (to mimic a large-scale setting)
        # repeats = args.scene_grid
        # gridx, gridy = torch.meshgrid(
        #     [
        #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        #     ],
        #     indexing="ij",
        # )
        # grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(
        #     -1, 3
        # )
        # means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
        # means = means.reshape(-1, 3)
        # quats = quats.repeat(repeats**2, 1)
        # scales = scales.repeat(repeats**2, 1)
        # colors = colors.repeat(repeats**2, 1, 1)
        # opacities = opacities.repeat(repeats**2)
        print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.if_cubemap:
            # fovx = 2 * torch.atan(width / 2.0 / K[0, 0])
            # fovy = 2 * torch.atan(height / 2.0 / K[1, 1])
            # print(K)
            # print(fovx, fovy)
            width = K[0, 0] * 2 * math.tan(1.57079632679 / 2)
            height = K[1, 1] * 2 * math.tan(1.57079632679 / 2)
            K[0, -1] = width / 2
            K[1, -1] = height / 2
            # fovx = 2 * torch.atan(width / 2.0 / K[0, 0])
            # fovy = 2 * torch.atan(height / 2.0 / K[1, 1])
            # print(K)
            # print(fovx, fovy)
            # import pdb; pdb.set_trace()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        if args.if_cubemap:
            camtoworlds_up = rotate_camera(viewmat, 90, 0, 0).unsqueeze(0)
            camtoworlds_down = rotate_camera(viewmat, -90, 0, 0).unsqueeze(0)
            camtoworlds_right = rotate_camera(viewmat, 0, 90, 0).unsqueeze(0)
            camtoworlds_left = rotate_camera(viewmat, 0, -90, 0).unsqueeze(0)
            camtoworlds_back = rotate_camera(viewmat, 0, 180, 0).unsqueeze(0)
            camtoworlds_duplicated = torch.cat([viewmat.unsqueeze(0), camtoworlds_up, camtoworlds_down, camtoworlds_left, camtoworlds_right, camtoworlds_back], dim=0)  # [6, 4, 4]
            Ks_duplicated = torch.cat([K.unsqueeze(0), K.unsqueeze(0), K.unsqueeze(0), K.unsqueeze(0), K.unsqueeze(0), K.unsqueeze(0)], dim=0)  # [6, 3, 3]
            packed = True
        else:
            camtoworlds_duplicated = viewmat[None]  # [1, 4, 4]
            Ks_duplicated = K[None]  # [1, 3, 3]
            packed = False

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            camtoworlds_duplicated,
            Ks_duplicated,
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
            packed=packed,
            if_cubemap=args.if_cubemap,
        )

        if args.if_cubemap:
            renders_forward = render_colors.narrow(0, 0, 1) 
            renders_up = render_colors.narrow(0, 1, 1)
            renders_down = render_colors.narrow(0, 2, 1)
            renders_left = render_colors.narrow(0, 3, 1)
            renders_right = render_colors.narrow(0, 4, 1)
            renders_back = render_colors.narrow(0, 5, 1)
            render_list = [renders_forward.squeeze(0).permute(2, 0, 1), 
                           renders_up.squeeze(0).permute(2, 0, 1), 
                           renders_down.squeeze(0).permute(2, 0, 1), 
                           renders_left.squeeze(0).permute(2, 0, 1), 
                           renders_right.squeeze(0).permute(2, 0, 1), 
                           renders_back.squeeze(0).permute(2, 0, 1)]
            render_rgbs = cubemap_to_panorama_torch(render_list, 0, step=0).permute(1, 2, 0).cpu().numpy()
        else:
            render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt results/garden/ckpts/ckpt_3499_rank1.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria")
    parser.add_argument("--if_cubemap", action="store_true", help="if use cubemap")

    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
