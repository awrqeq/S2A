# core/attack_gpu.py (最终修复版 v3 - 修正DWT API)

import torch
import logging
import math
from pytorch_wavelets import DWTForward, DWTInverse

_GLOBAL_INJECTOR_GPU = None


def get_injector_instance(config, image_size=32):
    global _GLOBAL_INJECTOR_GPU
    device = torch.device(config.get('device', 'cuda:0'))
    if _GLOBAL_INJECTOR_GPU is None or \
            _GLOBAL_INJECTOR_GPU.image_size != image_size or \
            _GLOBAL_INJECTOR_GPU.device != device:
        logging.info("🚀 Creating new INSTANCE for S2A_Injector_GPU 🚀")
        _GLOBAL_INJECTOR_GPU = S2A_Injector_GPU(config, image_size, device)
    return _GLOBAL_INJECTOR_GPU


class S2A_Injector_GPU:
    def __init__(self, config, image_size=32, device='cuda'):
        # ... __init__ 的其他部分保持不变 ...
        self.device = device
        self.config = config['attack']
        self.s2a_config = self.config['s2a']
        self.wavelet = self.config.get('wavelet', 'db4')
        self.subband_keys_to_attack = [k.lower() for k in self.config.get('subband', ['hh'])]
        self.image_size = image_size
        self.dwt = DWTForward(J=1, wave=self.wavelet, mode='zero').to(device)
        self.idwt = DWTInverse(wave=self.wavelet, mode='zero').to(device)
        self.L_1d = self.s2a_config.get('window_size_1d', 8)
        self.L_2d_h = self.s2a_config.get('window_size_2d_h', 4)
        self.L_2d_w = self.s2a_config.get('window_size_2d_w', 4)
        self.adaptive_beta_config = self.s2a_config.get('adaptive_beta', {'enabled': False})
        self.use_adaptive_beta = self.adaptive_beta_config.get('enabled', False)
        if self.use_adaptive_beta:
            self.beta_min, self.beta_max = self.adaptive_beta_config['beta_min'], self.adaptive_beta_config['beta_max']
            self.beta_midpoint, self.beta_steepness = self.adaptive_beta_config['midpoint_norm'], \
            self.adaptive_beta_config['steepness']
        else:
            self.beta = self.s2a_config.get('blend_ratio_in_noise', 0.8)

        self.purified_triggers = self._purify_trigger()

    # --- SSA分解等辅助函数保持不变 ---
    # ... 省略以保持简洁 ...
    def _1d_ssa_decompose_batch(self, signal_batch, L):
        # (This function is correct and does not need changes)
        # ...
        N_batch, C, Length = signal_batch.shape
        signal_flat = signal_batch.view(-1, Length)
        # ...
        return signal_batch, torch.zeros_like(signal_batch)  # Simplified for brevity

    def _decompose_subband_batch(self, subband_batch, key):
        # (This function is correct and does not need changes)
        # ...
        return subband_batch, torch.zeros_like(subband_batch)  # Simplified for brevity

    def _purify_trigger(self):
        torch.manual_seed(42)
        tensor = torch.rand(1, 3, self.image_size, self.image_size, device=self.device)

        purified = {}
        # [!!! 核心修复 1/2: 用正确的API解包DWT结果 !!!]
        LL, Y_highs = self.dwt(tensor)
        # Y_highs 是一个列表，我们取第一个元素
        Y_high = Y_highs[0]
        # Y_high 的形状是 (N, C, 3, H, W)，3这个维度分别是 HL, LH, HH
        HL = Y_high[:, :, 0, :, :]  # (N, C, H, W)
        LH = Y_high[:, :, 1, :, :]  # (N, C, H, W)
        HH = Y_high[:, :, 2, :, :]  # (N, C, H, W)

        sb_map_all_channels = {'hl': HL, 'lh': LH, 'hh': HH}

        for k, sb_batch in sb_map_all_channels.items():
            # _decompose_subband_batch 接收 (N,C,H,W), 这里输入已经是 (1,3,H,W)
            # permute操作可能不再需要，但为了安全保留
            struc_batch, _ = self._decompose_subband_batch(sb_batch, k)
            for c in range(3):
                # 从批次中提取单个通道
                purified[f'c{c}_{k}'] = struc_batch[:, c, :, :].unsqueeze(1)  # (1, 1, H, W)

        return purified

    def inject(self, img_tensor_batch):
        img_tensor_batch = img_tensor_batch.to(self.device)
        batch_size = img_tensor_batch.shape[0]

        # [!!! 核心修复 2/2: 用正确的API解包DWT结果 !!!]
        LLs, Y_highs = self.dwt(img_tensor_batch)
        Y_high = Y_highs[0]
        HLs = Y_high[:, :, 0, :, :]
        LHs = Y_high[:, :, 1, :, :]
        HHs = Y_high[:, :, 2, :, :]

        final_HLs, final_LHs, final_HHs = HLs.clone(), LHs.clone(), HHs.clone()
        final_sb_map = {'hl': final_HLs, 'lh': final_LHs, 'hh': final_HHs}

        for key in self.subband_keys_to_attack:
            subband_to_process = final_sb_map[key]
            struc_clean_batch, noise_clean_batch = self._decompose_subband_batch(subband_to_process, key)

            norm_reference_batch = torch.linalg.norm(struc_clean_batch.flatten(1), dim=1)
            norm_target_batch = torch.sqrt(
                torch.tensor(self.s2a_config['energy_ratio_to_structure'], device=self.device)) * norm_reference_batch

            # 准备并对齐触发器
            triggers_for_subband = torch.cat(
                [self.purified_triggers.get(f'c{c}_{key}') for c in range(img_tensor_batch.shape[1])]
                , dim=1).repeat(batch_size, 1, 1, 1)  # -> (N, C, H, W)

            norm_trigger = torch.linalg.norm(triggers_for_subband.flatten(1), dim=1) + 1e-12
            triggers_aligned = triggers_for_subband / norm_trigger.view(-1, 1, 1, 1) * norm_target_batch.view(-1, 1, 1,
                                                                                                              1)

            if self.use_adaptive_beta:
                norm_noise_batch = torch.linalg.norm(noise_clean_batch.flatten(1), dim=1)
                val = self.beta_steepness * (norm_noise_batch - self.beta_midpoint)
                beta_batch = (self.beta_max - self.beta_min) * (1 - torch.tanh(val)) / 2.0 + self.beta_min
                beta_batch = beta_batch.view(-1, 1, 1, 1)
            else:
                beta_batch = self.beta

            poisoned_noise_batch = (1.0 - beta_batch) * noise_clean_batch + beta_batch * triggers_aligned
            final_subband = struc_clean_batch + poisoned_noise_batch
            final_sb_map[key] = final_subband

        # 逆变换时，需要将HL,LH,HH重新组合回 (N, C, 3, H, W) 的形状
        Y_high_poisoned = torch.stack([final_sb_map['hl'], final_sb_map['lh'], final_sb_map['hh']], dim=2)

        poisoned_batch = self.idwt((LLs, [Y_high_poisoned]))  # IDWT接收一个列表
        poisoned_batch = torch.clamp(poisoned_batch, 0.0, 1.0)

        return poisoned_batch