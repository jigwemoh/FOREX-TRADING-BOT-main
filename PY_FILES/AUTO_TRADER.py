#!/usr/bin/env python3
"""
Automatic Trading Bot - Live Execution
Integrates ML predictions with MT5 live trading
"""

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    # Fallback to mock for development on non-Windows systems
    import MT5_MOCK as mt5  # type: ignore
import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib
import json
from typing import Dict, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../CSV_FILES/trading_log.txt'),
        logging.StreamHandler()
    ]
)

class AutoTrader:
    """Automated trading bot with ML predictions and risk management"""
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "1H",
        risk_percent: float = 1.0,
        max_positions: int = 3,
        use_ml: bool = True
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.use_ml = use_ml
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.is_running = False
        
        # MT5 timeframe mapping
        self.timeframe_map = {
            "1M": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "30M": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1
        }

    @staticmethod
    def _mt5_terminal_candidates() -> List[str]:
        """Common MT5 terminal locations on Windows VPS."""
        return [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files\MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
        ]
        
    def initialize_mt5(self, login: int, password: str, server: str, terminal_path: Optional[str] = None) -> bool:
        """Initialize MT5 connection"""
        init_ok = False
        init_args: List[Dict[str, str]] = []

        if terminal_path:
            init_args.append({"path": terminal_path})

        init_args.append({})

        for candidate in self._mt5_terminal_candidates():
            init_args.append({"path": candidate})

        for kwargs in init_args:
            try:
                if mt5.initialize(timeout=60000, **kwargs):
                    init_ok = True
                    if "path" in kwargs:
                        logging.info(f"MT5 initialized with terminal path: {kwargs['path']}")
                    else:
                        logging.info("MT5 initialized with default terminal lookup")
                    break
            except Exception as e:
                logging.warning(f"MT5 initialize attempt failed with args {kwargs}: {e}")

        if not init_ok:
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # If terminal is already authorized with the target account, use it.
        account_info = mt5.account_info()
        if account_info is not None and int(account_info.login) == int(login):
            logging.info(f"Connected to MT5 (existing session): {account_info}")
            return True
            
        if not mt5.login(login, password=password, server=server):
            logging.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
            
        logging.info(f"Connected to MT5: {mt5.account_info()}")
        return True
        
    def load_models(self) -> bool:
        """Load trained ML models and scalers"""
        if not self.use_ml:
            return True
            
        model_dir = Path("../ALL_MODELS") / self.symbol
        if not model_dir.exists():
            logging.warning(f"No models found for {self.symbol}, trading without ML")
            self.use_ml = False
            return True
            
        try:
            # Load models for different timeframes
            for tf in ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M"]:
                model_path = model_dir / f"{tf}.joblib"
                scaler_path = model_dir / f"{tf}_scaler.joblib"
                
                if model_path.exists() and scaler_path.exists():
                    self.models[tf] = joblib.load(model_path)
                    self.scalers[tf] = joblib.load(scaler_path)
                    logging.info(f"Loaded model: {tf}")
                    
            if not self.models:
                logging.warning("No models loaded, trading without ML")
                self.use_ml = False
                
            return True
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            self.use_ml = False
            return True
            
    def get_market_data(self, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch latest market data from MT5"""
        try:
            tf = self.timeframe_map.get(self.timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
            
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get market data: {mt5.last_error()}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            return df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return None
            
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for ML prediction"""
        try:
            # OHLC ratios
            df['HL_ratio'] = df['High'] / df['Low']
            df['OC_ratio'] = df['Open'] / df['Close']
            
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(5).mean()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            
            # Price relative to SMAs
            df['Close_SMA5'] = df['Close'] / df['SMA_5']
            df['Close_SMA20'] = df['Close'] / df['SMA_20']
            df['SMA5_SMA20'] = df['SMA_5'] / df['SMA_20']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            df['ATR_pct'] = df['ATR'] / df['Close']
            
            # Volume
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Fill NaN
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating features: {e}")
            return df
            
    def get_ml_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Get ML prediction signal
        Returns: (signal, confidence)
            signal: 1=BUY, -1=SELL, 0=HOLD
            confidence: 0.0-1.0
        """
        if not self.use_ml or not self.models:
            return 0, 0.0
            
        try:
            feature_cols = [
                'HL_ratio', 'OC_ratio', 'SMA_5', 'SMA_20', 'SMA_50',
                'Close_SMA5', 'Close_SMA20', 'SMA5_SMA20', 'RSI',
                'MACD', 'ATR', 'ATR_pct', 'Volume_SMA', 'Volume_ratio'
            ]
            
            # Get latest features
            X = df[feature_cols].iloc[-1:].values
            
            # Get predictions from all timeframe models
            predictions = []
            confidences = []
            
            for tf, model in self.models.items():
                scaler = self.scalers[tf]
                X_scaled = scaler.transform(X)
                
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                conf = prob.max()
                
                predictions.append(pred)
                confidences.append(conf)
                
            # Aggregate predictions (majority vote)
            avg_pred = np.mean(predictions)
            avg_conf = np.mean(confidences)
            
            # Signal logic
            if avg_pred > 0.6 and avg_conf > 0.55:  # Strong buy signal
                return 1, float(avg_conf)
            elif avg_pred < 0.4 and avg_conf > 0.55:  # Strong sell signal
                return -1, float(avg_conf)
            else:
                return 0, float(avg_conf)  # Hold
                
        except Exception as e:
            logging.error(f"Error getting ML signal: {e}")
            return 0, 0.0
            
    def get_smc_signal(self, df: pd.DataFrame) -> int:
        """
        Get Smart Money Concepts signal
        Returns: 1=BUY, -1=SELL, 0=HOLD
        """
        try:
            # Simple BOS detection
            lookback = 20
            recent = df.tail(lookback)
            
            current_high = recent['High'].iloc[-1]
            current_low = recent['Low'].iloc[-1]
            prev_high = recent['High'].iloc[-2:-1].max()
            prev_low = recent['Low'].iloc[-2:-1].min()
            
            # Bullish BOS (break above previous high)
            if current_high > prev_high:
                return 1
                
            # Bearish BOS (break below previous low)
            if current_low < prev_low:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error getting SMC signal: {e}")
            return 0
            
    def calculate_position_size(self, stop_loss_pips: float) -> float:
        """Calculate position size based on risk percentage"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return 0.01
                
            balance = account_info.balance
            risk_amount = balance * (self.risk_percent / 100)
            
            # Get symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                return 0.01
                
            tick_value = symbol_info.trade_tick_value
            
            # Calculate lot size
            pip_value = tick_value * 10  # Assuming 1 pip = 10 points
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to symbol's volume step
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Limit to min/max
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            
            return lot_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.01
            
    def get_open_positions(self) -> int:
        """Get number of open positions for this symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) if positions else 0
        
    def open_trade(self, signal: int, confidence: float, stop_loss_pips: float = 50, take_profit_pips: float = 100) -> bool:
        """
        Open a trade based on signal
        signal: 1=BUY, -1=SELL
        """
        try:
            # Check if max positions reached
            if self.get_open_positions() >= self.max_positions:
                logging.info(f"Max positions ({self.max_positions}) reached, skipping trade")
                return False
                
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logging.error("Failed to get current price")
                return False
                
            # Calculate position size
            lot_size = self.calculate_position_size(stop_loss_pips)
            
            # Prepare order
            symbol_info = mt5.symbol_info(self.symbol)
            point = symbol_info.point
            
            if signal == 1:  # BUY
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - stop_loss_pips * 10 * point
                tp = price + take_profit_pips * 10 * point
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + stop_loss_pips * 10 * point
                tp = price - take_profit_pips * 10 * point
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 234000,
                "comment": f"ML confidence: {confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result.comment}")
                return False
                
            logging.info(f"Order opened: {result.order} | Type: {'BUY' if signal == 1 else 'SELL'} | Lots: {lot_size} | Confidence: {confidence:.2f}")
            return True
            
        except Exception as e:
            logging.error(f"Error opening trade: {e}")
            return False
            
    def check_trailing_stop(self):
        """Check and update trailing stops for open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return
                
            for position in positions:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    continue
                    
                point = mt5.symbol_info(self.symbol).point
                trailing_stop = 30 * 10 * point  # 30 pips
                
                if position.type == mt5.POSITION_TYPE_BUY:
                    new_sl = tick.bid - trailing_stop
                    if new_sl > position.sl:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": self.symbol,
                            "position": position.ticket,
                            "sl": new_sl,
                            "tp": position.tp
                        }
                        mt5.order_send(request)
                        logging.info(f"Trailing stop updated for {position.ticket}: {new_sl}")
                        
                elif position.type == mt5.POSITION_TYPE_SELL:
                    new_sl = tick.ask + trailing_stop
                    if new_sl < position.sl or position.sl == 0:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": self.symbol,
                            "position": position.ticket,
                            "sl": new_sl,
                            "tp": position.tp
                        }
                        mt5.order_send(request)
                        logging.info(f"Trailing stop updated for {position.ticket}: {new_sl}")
                        
        except Exception as e:
            logging.error(f"Error checking trailing stop: {e}")
            
    def run(self, check_interval: int = 300):
        """
        Main trading loop
        check_interval: seconds between checks (default 5 minutes)
        """
        logging.info(f"Starting auto trader for {self.symbol}")
        logging.info(f"Risk: {self.risk_percent}% | Max positions: {self.max_positions} | ML: {self.use_ml}")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Get market data
                df = self.get_market_data(bars=100)
                if df is None:
                    time.sleep(60)
                    continue
                    
                # Calculate features
                df = self.calculate_features(df)
                
                # Get signals
                ml_signal, ml_confidence = self.get_ml_signal(df)
                smc_signal = self.get_smc_signal(df)
                
                logging.info(f"ML Signal: {ml_signal} (conf: {ml_confidence:.2f}) | SMC Signal: {smc_signal}")
                
                # Combined signal logic
                if self.use_ml:
                    # Use ML signal if confidence is high
                    if ml_signal != 0 and ml_confidence > 0.6:
                        self.open_trade(ml_signal, ml_confidence)
                else:
                    # Use SMC signal
                    if smc_signal != 0:
                        self.open_trade(smc_signal, 0.5)
                        
                # Check trailing stops
                self.check_trailing_stop()
                
                # Wait for next check
                logging.info(f"Waiting {check_interval}s for next check...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logging.info("Auto trader stopped by user")
            self.stop()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            self.stop()
            
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        mt5.shutdown()
        logging.info("Auto trader stopped")


if __name__ == "__main__":
    config_path = Path("../config.json")
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Run: python CONFIG_MANAGER.py create")
        exit(1)

    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
    except Exception as error:
        print(f"Failed to load configuration: {error}")
        exit(1)

    mt5_cfg = config.get("mt5", {})
    trading_cfg = config.get("trading", {})
    execution_cfg = config.get("execution", {})

    MT5_LOGIN = int(mt5_cfg.get("login", 0))
    MT5_PASSWORD = str(mt5_cfg.get("password", ""))
    MT5_SERVER = str(mt5_cfg.get("server", ""))
    MT5_TERMINAL_PATH = str(mt5_cfg.get("terminal_path", "")).strip() or None

    if MT5_LOGIN == 0 or not MT5_PASSWORD or not MT5_SERVER:
        print("Invalid MT5 configuration in ../config.json")
        print("Run: python CONFIG_MANAGER.py create")
        exit(1)

    SYMBOL = str(trading_cfg.get("symbol", "EURUSD"))
    TIMEFRAME = str(trading_cfg.get("timeframe", "1H")).upper()
    RISK_PERCENT = float(trading_cfg.get("risk_percent", 1.0))
    MAX_POSITIONS = int(trading_cfg.get("max_positions", 3))
    USE_ML = bool(trading_cfg.get("use_ml", True))
    CHECK_INTERVAL = int(execution_cfg.get("check_interval", 300))
    
    # Create trader
    trader = AutoTrader(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        risk_percent=RISK_PERCENT,
        max_positions=MAX_POSITIONS,
        use_ml=USE_ML
    )
    
    # Initialize MT5
    if not trader.initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH):
        print("Failed to initialize MT5")
        exit(1)
        
    # Load models
    if not trader.load_models():
        print("Failed to load models")
        exit(1)
        
    # Start trading
    trader.run(check_interval=CHECK_INTERVAL)
