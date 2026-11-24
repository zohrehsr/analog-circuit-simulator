# راهنمای اجرای اولیه پروژه

## ۱. فعال‌سازی venv (ساخته شده توسط uv)
```powershell
cd C:\Users\Zohreh\.gemini\antigravity\scratch\analog-circuit-simulator
.\.venv\Scripts\activate.ps1
```

## ۲. نصب وابستگی‌ها (با اتصال اینترنت عادی)
```powershell
pip install -e ".[dev]"
```

## ۳. اجرای تست‌ها
```powershell
pytest -v
```

## ۴. اجرای مثال‌ها
```powershell
python examples/rc_lowpass_filter.py
python examples/rlc_bandpass_filter.py
```

## ۵. استفاده از CLI
```powershell
analog-sim analyze --type rc --R 1000 --C 1e-6
analog-sim simulate --type rlc --R 100 --L 10e-3 --C 1e-6
```

## ۶. راه‌اندازی GitHub ریپو
```powershell
git init
git add .
git commit -m "feat: initial release v0.1.0 - RC/RL/RLC circuit simulator"
# بعد در GitHub یک ریپو بسازید و:
git remote add origin https://github.com/YOUR_USERNAME/analog-circuit-simulator.git
git push -u origin main
```
