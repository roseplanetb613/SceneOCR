@echo off
REM ===== 检测长训一键启动 =====
REM 作用: 用自己的 Hiera+FPN+DBNetHead 在 SynthText 上大规模训练
REM 用法: 双击运行; 想停止就按 Ctrl+C(会自动保存断点, 下次双击自动续训)
REM 可改: --steps 训练步数, --synthtext_max 数据子集大小, --ckpt 断点路径
setlocal
set PYTHONPATH=%~dp0
python examples\pretrain_detector.py ^
  --data synthtext ^
  --synthtext_max 100000 ^
  --img_size 640 ^
  --batch 2 ^
  --train_backbone ^
  --pretrained_backbone E:\Sam2\sam2\checkpoints\sam2.1_hiera_tiny.pt ^
  --steps 30000 ^
  --log_every 200 ^
  --save_every 1000 ^
  --ckpt checkpoints\det_long.pt ^
  --log logs\det_long.log ^
  --resume
pause
