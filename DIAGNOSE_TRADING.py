#!/usr/bin/env python3
"""
Trading System Diagnostic Tool
Checks why no trades are being executed
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / "PY_FILES"))

def check_config():
    """Check if config.json has all required fields"""
    print("\n" + "="*80)
    print("1. CONFIG FILE CHECK")
    print("="*80)
    
    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ config.json NOT FOUND")
        return False
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"✓ config.json exists")
    print(f"\nCurrent config:")
    print(json.dumps(config, indent=2))
    
    # Check required fields for AUTO_TRADER_MULTI.py
    required = {
        "mt5": ["login", "password", "server"],
        "trading": ["timeframe", "risk_percent", "max_positions", "use_ml", "ml_threshold"],
        "execution": ["check_interval"]
    }
    
    missing = []
    for section, fields in required.items():
        if section not in config:
            missing.append(f"{section} section")
            continue
        for field in fields:
            if field not in config[section]:
                missing.append(f"{section}.{field}")
    
    if missing:
        print(f"\n❌ Missing required fields:")
        for m in missing:
            print(f"   - {m}")
        return False
    else:
        print(f"\n✓ All required fields present")
        return True

def check_models():
    """Check if models exist and are loadable"""
    print("\n" + "="*80)
    print("2. MODEL AVAILABILITY CHECK")
    print("="*80)
    
    models_dir = Path("ALL_MODELS")
    if not models_dir.exists():
        print("❌ ALL_MODELS directory not found")
        return False
    
    symbols = []
    for item in sorted(models_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        
        has_models = any(item.glob("T_*.joblib"))
        has_features = (item / "features.joblib").exists()
        has_metadata = (item / "model_metadata.json").exists()
        
        if has_models and has_features:
            symbols.append(item.name)
            status = "✓"
            if has_metadata:
                with open(item / "model_metadata.json") as f:
                    meta = json.load(f)
                    threshold = meta.get("optimal_threshold", meta.get("1H", {}).get("optimal_threshold", 0.5))
                    print(f"{status} {item.name:12} | Models: {len(list(item.glob('T_*.joblib')))} | Threshold: {threshold:.2f}")
            else:
                print(f"{status} {item.name:12} | Models: {len(list(item.glob('T_*.joblib')))} | No metadata")
        else:
            print(f"  {item.name:12} | Missing models or features")
    
    if not symbols:
        print("❌ No valid model folders found")
        return False
    
    print(f"\n✓ Found {len(symbols)} tradeable symbols: {', '.join(symbols[:5])}...")
    return True

def check_mt5_connection():
    """Test MT5 connection"""
    print("\n" + "="*80)
    print("3. MT5 CONNECTION CHECK")
    print("="*80)
    
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 module imported")
    except ImportError:
        print("❌ MetaTrader5 module not installed (expected on non-Windows)")
        print("   Using mock for testing")
        return None
    
    # Try to initialize without login
    if not mt5.initialize():
        print(f"⚠ MT5 not initialized: {mt5.last_error()}")
        print("   (This is expected if MT5 terminal is not running)")
        return False
    
    print("✓ MT5 initialized")
    
    account_info = mt5.account_info()
    if account_info:
        print(f"✓ Connected to account: {account_info.login}")
        print(f"  Balance: ${account_info.balance:.2f}")
        print(f"  Equity: ${account_info.equity:.2f}")
        print(f"  Margin Free: ${account_info.margin_free:.2f}")
        
        # Check open positions
        positions = mt5.positions_total()
        print(f"  Open Positions: {positions}")
        
        return True
    else:
        print("⚠ No account connected")
        return False

def check_signal_generation():
    """Test if signals can be generated"""
    print("\n" + "="*80)
    print("4. SIGNAL GENERATION CHECK")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        import joblib
        
        # Load EURUSD model as test
        model_path = Path("ALL_MODELS/EURUSD/T_1H.joblib")
        scaler_path = Path("ALL_MODELS/EURUSD/T_1H_scaler.joblib")
        features_path = Path("ALL_MODELS/EURUSD/features.joblib")
        
        if not all([model_path.exists(), scaler_path.exists(), features_path.exists()]):
            print("❌ EURUSD model files not found")
            return False
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        
        print(f"✓ Loaded EURUSD 1H model")
        print(f"  Required features: {len(features)}")
        
        # Create dummy data
        dummy_data = pd.DataFrame({feat: [0.5] for feat in features})
        dummy_scaled = scaler.transform(dummy_data)
        
        # Test prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(dummy_scaled)[0]
            print(f"  Prediction test: BUY={proba[1]:.3f}, SELL={proba[0]:.3f}")
            print(f"  Max confidence: {max(proba):.3f}")
            
            # Check threshold
            meta_path = Path("ALL_MODELS/EURUSD/model_metadata.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    threshold = meta.get("1H", {}).get("optimal_threshold", 0.55)
                    print(f"  Optimal threshold: {threshold:.2f}")
                    
                    if max(proba) >= threshold:
                        print(f"  ✓ Dummy signal would PASS threshold")
                    else:
                        print(f"  ⚠ Dummy signal would FAIL threshold")
        else:
            print("  ⚠ Model doesn't have predict_proba method")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing signal generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("FOREX TRADING BOT - SYSTEM DIAGNOSTIC")
    print("="*80)
    
    checks = [
        ("Config", check_config),
        ("Models", check_models),
        ("MT5 Connection", check_mt5_connection),
        ("Signal Generation", check_signal_generation),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, result in results.items():
        status = "✓" if result else "❌" if result is False else "⚠"
        print(f"{status} {name}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if not results.get("Config"):
        print("\n1. FIX CONFIG FILE:")
        print("   Run: python CONFIG_MANAGER.py create")
        print("   Or manually update config.json with required fields")
    
    if not results.get("Models"):
        print("\n2. TRAIN MODELS:")
        print("   Run: python PY_FILES/TRAIN_XGBOOST_FAST.py")
    
    if results.get("MT5 Connection") is False:
        print("\n3. START MT5 TERMINAL:")
        print("   - Launch MetaTrader 5 application")
        print("   - Login to your account")
        print("   - Then restart the trading bot")
    
    if all([results.get("Config"), results.get("Models")]):
        print("\n✓ SYSTEM READY FOR TRADING")
        print("\nTo start trading:")
        print("   python PY_FILES/AUTO_TRADER_MULTI.py")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
