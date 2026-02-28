#!/usr/bin/env python3
"""
Automatic Trading Bot - Multi-Symbol Live Execution
Integrates ML predictions with MT5 live trading across multiple symbols (forex + crypto)
"""

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    # Fallback to mock for development on non-Windows systems
    import MT5_MOCK as mt5  # type: ignore
import pandas as pd
from pathlib import Path
import time
import joblib
import json
from typing import Dict, Tuple, Optional, List
import logging
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../CSV_FILES/trading_log.txt'),
        logging.StreamHandler()
    ]
)

class MultiSymbolAutoTrader:
    """Automated trading bot with ML predictions and risk management for multiple symbols"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = "1H",
        risk_percent: float = 1.0,
        max_positions: int = 3,
        use_ml: bool = True
    ):
        self.symbols = symbols or ["EURUSD"]
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.use_ml = use_ml
        self.models: Dict[str, Dict[str, object]] = {sym: {} for sym in self.symbols}
        self.scalers: Dict[str, Dict[str, object]] = {sym: {} for sym in self.symbols}
        self.is_running = False
        
        # MT5 timeframe mapping
        self.timeframe_map = {
            "1M": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M5,
            "10M": mt5.TIMEFRAME_M10,
            "15M": mt5.TIMEFRAME_M15,
            "20M": mt5.TIMEFRAME_M20,
            "30M": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1,
            "1W": mt5.TIMEFRAME_W1,
            "1M_tf": mt5.TIMEFRAME_MN1
        }
        
        # Major crypto pairs support
        self.crypto_pairs = {"BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "ADAUSD", "SOLUSD", "DOGEUSD"}
        self.forex_pairs = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"}

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
        """Load trained ML models and scalers for all symbols"""
        if not self.use_ml:
            return True
        
        loaded_count = 0
        for symbol in self.symbols:
            model_dir = Path("../ALL_MODELS") / symbol
            if not model_dir.exists():
                logging.warning(f"No models found for {symbol}")
                continue
                
            try:
                # Load models for different timeframes (intraday and daily/higher)
                timeframes = ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M", "T_1H", "T_4H", "T_1D", "T_1W", "T_1M"]
                for tf in timeframes:
                    model_path = model_dir / f"{tf}.joblib"
                    scaler_path = model_dir / f"{tf}_scaler.joblib"
                    
                    if model_path.exists() and scaler_path.exists():
                        self.models[symbol][tf] = joblib.load(model_path)
                        self.scalers[symbol][tf] = joblib.load(scaler_path)
                        loaded_count += 1
                        logging.info(f"Loaded model: {symbol}/{tf}")
                        
            except Exception as e:
                logging.error(f"Error loading models for {symbol}: {e}")
        
        if loaded_count == 0 and self.use_ml:
            logging.warning("No models loaded, trading without ML")
            self.use_ml = False
            
        return True
            
    def get_market_data(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch latest market data from MT5 for a specific symbol"""
        try:
            # Select symbol in terminal
            if not mt5.symbol_select(symbol, True):
                logging.warning(f"Could not select {symbol}, attempting anyway...")
            
            tf = self.timeframe_map.get(self.timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get market data for {symbol}: {mt5.last_error()}")
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
            logging.error(f"Error fetching market data for {symbol}: {e}")
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
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['BB_upper'] = sma + (std * 2)
            df['BB_lower'] = sma - (std * 2)
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating features: {e}")
            return df
            
    def get_ml_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[int, float]:
        """Get ML prediction signal for a symbol"""
        if not self.use_ml or symbol not in self.models or not self.models[symbol]:
            return 0, 0.0
            
        try:
            # Use most recent features
            features = df.dropna().iloc[-1:][['HL_ratio', 'OC_ratio', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_signal']]
            
            if len(features) == 0:
                return 0, 0.0
            
            # Get the first available model and scaler
            if self.models[symbol]:
                first_tf = list(self.models[symbol].keys())[0]
                model = self.models[symbol][first_tf]
                scaler = self.scalers[symbol][first_tf]
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                confidence = abs(prediction)
                signal = 1 if prediction > 0 else -1 if prediction < 0 else 0
                
                return signal, confidence
                
            return 0, 0.0
            
        except Exception as e:
            logging.warning(f"Error getting ML signal for {symbol}: {e}")
            return 0, 0.0
            
    def get_smc_signal(self, df: pd.DataFrame) -> int:
        """Get Smart Money Concept signal based on price structure"""
        if len(df) < 20:
            return 0
            
        try:
            recent = df.iloc[-20:].copy()
            
            # Higher high and higher low = bullish
            if recent['High'].iloc[-1] > recent['High'].iloc[-2] and recent['Low'].iloc[-1] > recent['Low'].iloc[-2]:
                return 1
            # Lower high and lower low = bearish
            elif recent['High'].iloc[-1] < recent['High'].iloc[-2] and recent['Low'].iloc[-1] < recent['Low'].iloc[-2]:
                return -1
            else:
                return 0
                
        except Exception as e:
            logging.warning(f"Error getting SMC signal: {e}")
            return 0
            
    def get_open_positions(self, symbol: str) -> int:
        """Get number of open positions for a specific symbol"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            return len(positions) if positions else 0
        except Exception as e:
            logging.warning(f"Error getting positions for {symbol}: {e}")
            return 0
            
    def open_trade(self, symbol: str, signal: int, confidence: float):
        """Open a trade for a specific symbol"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Cannot get info for {symbol}")
                return
                
            tick_value = symbol_info.trade_tick_value
            
            # Calculate lot size based on risk
            account = mt5.account_info()
            risk_money = account.balance * (self.risk_percent / 100)
            lot_size = risk_money / (50 * tick_value)  # 50 pips stop loss
            
            # Round to symbol's volume step
            lot_step = symbol_info.volume_step
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Get market data for entry
            df = self.get_market_data(symbol, 100)
            if df is None:
                return
                
            current_price = df['Close'].iloc[-1]
            
            # Create order
            if signal == 1:
                order_type = mt5.ORDER_TYPE_BUY
                order_comment = f"ML_BUY_{confidence:.2f}"
            else:
                order_type = mt5.ORDER_TYPE_SELL
                order_comment = f"ML_SELL_{confidence:.2f}"
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": current_price,
                "sl": current_price - (50 * symbol_info.point) if signal == 1 else current_price + (50 * symbol_info.point),
                "tp": current_price + (100 * symbol_info.point) if signal == 1 else current_price - (100 * symbol_info.point),
                "comment": order_comment,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{symbol}] Trade opened: {order_comment} | Lot: {lot_size}")
            else:
                logging.error(f"[{symbol}] Trade failed: {result.comment}")
                
        except Exception as e:
            logging.error(f"Error opening trade for {symbol}: {e}")
            
    def check_trailing_stop(self, symbol: str):
        """Check and update trailing stops for a symbol"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return
                
            for pos in positions:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    continue
                
                current_price = mt5.symbol_info_tick(symbol).ask if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
                
                # Trailing stop logic (simplified)
                if pos.type == mt5.ORDER_TYPE_BUY and current_price > pos.price_open + (30 * symbol_info.point):
                    new_sl = current_price - (30 * symbol_info.point)
                    if new_sl > pos.sl:
                        # Update stop loss
                        pass
                        
        except Exception as e:
            logging.warning(f"Error checking trailing stop for {symbol}: {e}")
            
    def run_symbol(self, symbol: str, check_interval: int):
        """Trading loop for a specific symbol"""
        logging.info(f"[{symbol}] Starting trader...")
        
        while self.is_running:
            try:
                # Get market data
                df = self.get_market_data(symbol, 100)
                if df is None:
                    time.sleep(60)
                    continue
                    
                # Calculate features
                df = self.calculate_features(df)
                
                # Get signals
                ml_signal, ml_confidence = self.get_ml_signal(symbol, df)
                smc_signal = self.get_smc_signal(df)
                
                logging.info(f"[{symbol}] ML Signal: {ml_signal} (conf: {ml_confidence:.2f}) | SMC Signal: {smc_signal}")
                
                # Combined signal logic
                open_positions = self.get_open_positions(symbol)
                if open_positions < self.max_positions:
                    if self.use_ml:
                        if ml_signal != 0 and ml_confidence > 0.6:
                            self.open_trade(symbol, ml_signal, ml_confidence)
                    else:
                        if smc_signal != 0:
                            self.open_trade(symbol, smc_signal, 0.5)
                
                # Check trailing stops
                self.check_trailing_stop(symbol)
                
                # Wait for next check
                time.sleep(check_interval)
                
            except Exception as e:
                logging.error(f"[{symbol}] Error in main loop: {e}")
                time.sleep(60)
                
    def run(self, check_interval: int = 300):
        """
        Main trading loop for multiple symbols
        check_interval: seconds between checks (default 5 minutes)
        """
        logging.info(f"Starting multi-symbol auto trader")
        logging.info(f"Symbols: {', '.join(self.symbols)}")
        logging.info(f"Risk: {self.risk_percent}% | Max positions per symbol: {self.max_positions} | ML: {self.use_ml}")
        
        self.is_running = True
        threads = []
        
        try:
            # Start a trading thread for each symbol
            for symbol in self.symbols:
                thread = Thread(target=self.run_symbol, args=(symbol, check_interval), daemon=True)
                thread.start()
                threads.append(thread)
                logging.info(f"Started trading thread for {symbol}")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(1)
                
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

    # Support both single symbol (legacy) and multiple symbols
    if "symbols" in trading_cfg:
        symbols_list = trading_cfg.get("symbols", ["EURUSD"])
    else:
        # Fallback for old config format
        symbols_list = [trading_cfg.get("symbol", "EURUSD")]
    
    TIMEFRAME = str(trading_cfg.get("timeframe", "1H")).upper()
    RISK_PERCENT = float(trading_cfg.get("risk_percent", 1.0))
    MAX_POSITIONS = int(trading_cfg.get("max_positions", 3))
    USE_ML = bool(trading_cfg.get("use_ml", True))
    CHECK_INTERVAL = int(execution_cfg.get("check_interval", 300))
    
    # Create trader
    trader = MultiSymbolAutoTrader(
        symbols=symbols_list,
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
