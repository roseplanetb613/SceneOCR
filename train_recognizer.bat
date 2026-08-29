@echo off
REM ==== Recognition long training launcher ====
REM Double-click to start. Press Ctrl+C to stop (checkpoint auto-saved).
REM Double-click again to resume automatically.
REM 62-char vocab (digits + upper + lower). ckpt = synth pretrain (resume to continue).
REM For real-data fine-tune, run pretrain_ctc.py --data synthtext_rec manually
REM (see README quick start).
setlocal
set PYTHONPATH=%~dp0
python examples\pretrain_ctc.py ^
  --alphabet "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" ^
  --fixed_size 10000 ^
  --batch 32 ^
  --steps 20000 ^
  --log_every 500 ^
  --save_every 2000 ^
  --ckpt checkpoints\ctc_synth62.pt ^
  --log logs\ctc_train.log ^
  --resume
pause
