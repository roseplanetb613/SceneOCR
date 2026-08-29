@echo off
REM ==== Detection long training launcher ====
REM Double-click to start. Press Ctrl+C to stop (checkpoint auto-saved).
REM Double-click again to resume automatically.
REM Edit --steps / --synthtext_max / --ckpt as needed.
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
