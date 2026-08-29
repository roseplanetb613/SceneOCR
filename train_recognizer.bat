@echo off
REM ===== 识别长训一键启动 =====
REM 作用: 用自己的 CTCHead 在合成文本行上训练(已收敛到 94.5%, 可加大数据继续提精度)
REM 用法: 双击运行; 想停止就按 Ctrl+C(会自动保存断点, 下次双击自动续训)
setlocal
set PYTHONPATH=%~dp0
python examples\pretrain_ctc.py ^
  --alphabet "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" ^
  --fixed_size 10000 ^
  --batch 32 ^
  --steps 20000 ^
  --log_every 500 ^
  --save_every 2000 ^
  --ckpt checkpoints\ctc_full.pt ^
  --log logs\ctc_train.log ^
  --resume
pause
