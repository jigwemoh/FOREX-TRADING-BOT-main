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
import numpy as np
from pathlib import Path
import time
import joblib
import json
from typing import Dict, Tuple, Optional, List, Any, Set
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
        use_ml: bool = True,
        ml_threshold: float = 0.55
    ):
        self.symbols = symbols or ["EURUSD"]
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.use_ml = use_ml
        self.ml_threshold = ml_threshold
        self.models: Dict[str, Dict[str, object]] = {sym: {} for sym in self.symbols}
        self.scalers: Dict[str, Dict[str, object]] = {sym: {} for sym in self.symbols}
        self.model_features: Dict[str, List[str]] = {sym: [] for sym in self.symbols}
        self._feature_gap_logged: Set[str] = set()
        self._ml_diag_logged_symbols: Set[str] = set()
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

    @staticmethod
    def discover_symbols_from_models(models_root: Path = Path("../ALL_MODELS")) -> List[str]:
        """Discover tradable symbols from model folders only."""
        if not models_root.exists():
            return []

        symbols: List[str] = []
        for item in sorted(models_root.iterdir()):
            if not item.is_dir():
                continue
            if item.name.startswith("."):
                continue

            has_model_files = any(item.glob("T_*.joblib"))
            has_feature_file = (item / "features.joblib").exists()
            if has_model_files and has_feature_file:
                symbols.append(item.name)

        return symbols

    def filter_symbols_by_mt5_availability(self) -> Tuple[List[str], List[str]]:
        """Keep only symbols that are available in the connected MT5 terminal."""
        available: List[str] = []
        skipped: List[str] = []

        for symbol in self.symbols:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    skipped.append(symbol)
                    continue

                if not mt5.symbol_select(symbol, True):
                    skipped.append(symbol)
                    continue

                available.append(symbol)
            except Exception:
                skipped.append(symbol)

        self.symbols = available
        self.models = {sym: {} for sym in self.symbols}
        self.scalers = {sym: {} for sym in self.symbols}
        self.model_features = {sym: [] for sym in self.symbols}
        self._feature_gap_logged = set()
        self._ml_diag_logged_symbols = set()

        return available, skipped
        
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
                features_path = model_dir / "features.joblib"
                if features_path.exists():
                    loaded_features = joblib.load(features_path)
                    if isinstance(loaded_features, list):
                        self.model_features[symbol] = [str(feature) for feature in loaded_features]
                        logging.info(
                            f"Loaded feature metadata: {symbol} ({len(self.model_features[symbol])} features)"
                        )
                    else:
                        logging.warning(f"Invalid features list for {symbol}, using empty list")

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
            df = df.copy()

            # OHLC ratios
            df['HL_ratio'] = df['High'] / df['Low']
            df['OC_ratio'] = df['Open'] / df['Close']
            df['HL_Ratio'] = df['HL_ratio']
            df['OC_Ratio'] = df['OC_ratio']
            
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(5).mean()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Price relative to SMAs
            df['Close_SMA5'] = df['Close'] / df['SMA_5']
            df['Close_SMA20'] = df['Close'] / df['SMA_20']
            df['SMA5_SMA20'] = df['SMA_5'] / df['SMA_20']
            df['Range'] = df['High'] - df['Low']
            df['Body'] = (df['Close'] - df['Open']).abs()
            df['Body_Range'] = df['Body'] / df['Range'].replace(0, np.nan)
            
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
            df['Signal'] = df['MACD_signal']
            df['Histogram'] = df['MACD_hist']
            
            # Bollinger Bands
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['BB_upper'] = sma + (std * 2)
            df['BB_lower'] = sma - (std * 2)
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            df['BB_Middle'] = sma
            df['BB_Std'] = std
            df['BB_Upper'] = df['BB_upper']
            df['BB_Lower'] = df['BB_lower']

            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift(1)).abs()
            tr3 = (df['Low'] - df['Close'].shift(1)).abs()
            df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))
            df['ATR'] = df['TR'].rolling(14).mean()

            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, np.nan)

            df['Trend'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
            df['Price_Above_SMA'] = (df['Close'] > df['SMA_50']).astype(int)

            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Return_MA'] = df['Log_Return'].rolling(5).mean()
            df['Return_Std'] = df['Log_Return'].rolling(5).std()

            df = df.replace([np.inf, -np.inf], np.nan)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating features: {e}")
            return df
            
    def get_ml_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[int, float]:
        """Get ML prediction signal for a symbol"""
        if not self.use_ml or symbol not in self.models or not self.models[symbol]:
            return 0, 0.0
            
        try:
            target_tf_map = {
                "5M": "T_5M",
                "10M": "T_10M",
                "15M": "T_15M",
                "20M": "T_20M",
                "30M": "T_30M",
                "1H": "T_1H",
                "4H": "T_4H",
                "1D": "T_1D",
                "1W": "T_1W",
                "1M_TF": "T_1M",
                "1M": "T_1M",
            }

            target_tf = target_tf_map.get(self.timeframe.upper(), "T_1H")
            model = self.models[symbol].get(target_tf)
            scaler = self.scalers[symbol].get(target_tf)

            if model is None or scaler is None:
                first_tf = next(iter(self.models[symbol]), None)
                if first_tf is None:
                    return 0, 0.0
                model = self.models[symbol][first_tf]
                scaler = self.scalers[symbol][first_tf]

            required_features = self.model_features.get(symbol, [])
            if not required_features:
                logging.warning(f"No saved feature list for {symbol}")
                return 0, 0.0

            diag_key = f"{symbol}:{target_tf}"
            if diag_key not in self._ml_diag_logged_symbols:
                logging.info(
                    f"[ML CONFIG] {symbol} | timeframe={self.timeframe} ({target_tf}) | "
                    f"threshold={self.ml_threshold:.2f} | required_features={len(required_features)}"
                )
                self._ml_diag_logged_symbols.add(diag_key)

            latest = df.dropna().iloc[-1:]
            if latest.empty:
                return 0, 0.0

            missing_features = [feature for feature in required_features if feature not in latest.columns]
            for feature in missing_features:
                latest[feature] = 0.0

            if len(missing_features) > 0 and diag_key not in self._feature_gap_logged:
                sample_missing = ", ".join(missing_features[:8])
                suffix = " ..." if len(missing_features) > 8 else ""
                logging.warning(
                    f"[ML FEATURE GAP] {symbol} missing {len(missing_features)} features; "
                    f"filled with 0.0 | sample: {sample_missing}{suffix}"
                )
                self._feature_gap_logged.add(diag_key)

            features = latest[required_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

            if features.empty:
                return 0, 0.0

            features_scaled = scaler.transform(features)

            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(features_scaled))[0]
                if probabilities.shape[0] < 2:
                    return 0, 0.0

                prob_down = float(probabilities[0])
                prob_up = float(probabilities[1])
                confidence = max(prob_up, prob_down)

                if prob_up >= self.ml_threshold and prob_up > prob_down:
                    return 1, confidence
                if prob_down >= self.ml_threshold and prob_down > prob_up:
                    return -1, confidence

                return 0, confidence

            prediction = int(model.predict(features_scaled)[0])
            if prediction == 1:
                return 1, 0.5
            if prediction == 0:
                return -1, 0.5
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
    
    def get_all_open_positions(self) -> List[Any]:
        """Get ALL open positions on the MT5 account (bot and manual trades)"""
        try:
            all_positions = mt5.positions_get()
            return list(all_positions) if all_positions else []
        except Exception as e:
            logging.warning(f"Error getting all positions: {e}")
            return []
    
    def manage_all_trades(self):
        """Manage ALL open trades on the account (bot-generated and manually opened)
        
        This includes:
        - Applying trailing stops
        - Checking risk thresholds
        - Auto-closing based on risk management rules
        - Logging all trade activity
        """
        try:
            all_positions = self.get_all_open_positions()
            if not all_positions:
                return
            
            logging.info(f"Managing {len(all_positions)} total open position(s) on account")
            
            for pos in all_positions:
                symbol = pos.symbol
                position_id = pos.ticket
                
                try:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info is None:
                        logging.warning(f"Cannot get info for {symbol}, skipping position {position_id}")
                        continue
                    
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        logging.warning(f"Cannot get tick for {symbol}, skipping position {position_id}")
                        continue
                    
                    current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                    profit_points = (current_price - pos.price_open) if pos.type == mt5.ORDER_TYPE_BUY else (pos.price_open - current_price)
                    profit_pips = profit_points / symbol_info.point
                    profit_loss = pos.profit
                    
                    # Log position status
                    pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                    logging.info(
                        f"[MANAGE] {symbol} {pos_type} | Ticket: {position_id} | "
                        f"Entry: {pos.price_open:.5f} | Current: {current_price:.5f} | "
                        f"Profit: {profit_pips:.1f}pips ({profit_loss:.2f}) | "
                        f"SL: {pos.sl} | TP: {pos.tp}"
                    )
                    
                    # Apply trailing stop
                    self._update_trailing_stop(pos, symbol_info, current_price)
                    
                    # Check if position should be closed based on risk/profit
                    self._check_position_close_conditions(pos, symbol_info, current_price, profit_loss, profit_pips)
                    
                except Exception as e:
                    logging.error(f"Error managing position {position_id}: {e}")
                    
        except Exception as e:
            logging.error(f"Error in manage_all_trades: {e}")
    
    def _update_trailing_stop(self, pos: Any, symbol_info: Any, current_price: float):
        """Update trailing stop loss for a position (applies to all trades)"""
        try:
            trailing_stop_pips = 30  # Configurable via config.json if needed
            trailing_stop_distance = trailing_stop_pips * symbol_info.point
            
            if pos.type == mt5.ORDER_TYPE_BUY:
                # For buy positions, trailing stop moves up but never down
                new_sl = current_price - trailing_stop_distance
                if new_sl > pos.sl:
                    self._modify_position_sl(pos, new_sl, symbol_info)
                    
            elif pos.type == mt5.ORDER_TYPE_SELL:
                # For sell positions, trailing stop moves down but never up
                new_sl = current_price + trailing_stop_distance
                if new_sl < pos.sl:
                    self._modify_position_sl(pos, new_sl, symbol_info)
                    
        except Exception as e:
            logging.warning(f"Error updating trailing stop for ticket {pos.ticket}: {e}")
    
    def _modify_position_sl(self, pos: Any, new_sl: float, symbol_info: Any):
        """Modify stop loss for a position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "sl": new_sl,
                "tp": pos.tp,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[TRAILING STOP] Ticket {pos.ticket}: Updated SL to {new_sl:.5f}")
            else:
                logging.warning(f"Failed to update SL for ticket {pos.ticket}: {result.comment}")
                
        except Exception as e:
            logging.error(f"Error modifying SL for ticket {pos.ticket}: {e}")
    
    def _check_position_close_conditions(self, pos: Any, symbol_info: Any, current_price: float, profit_loss: float, profit_pips: float):
        """Check if a position should be closed based on conditions
        
        Conditions checked:
        - Stop loss hit (should auto-close, but double-check)
        - Take profit hit (should auto-close, but double-check)
        - Excessive drawdown on account (risk management override)
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return
            
            account_balance = account_info.balance
            max_account_drawdown_percent = 5.0  # Configurable
            max_drawdown = account_balance * (max_account_drawdown_percent / 100)
            
            # Check if account is in excessive drawdown
            if account_info.profit < -max_drawdown:
                logging.warning(
                    f"Account drawdown excessive: {account_info.profit:.2f} (limit: {-max_drawdown:.2f}). "
                    f"Closing position {pos.ticket}"
                )
                self._close_position(pos)
                return
            
            # Close if hit stop loss (safety check)
            if pos.sl > 0 and (
                (pos.type == mt5.ORDER_TYPE_BUY and current_price <= pos.sl) or
                (pos.type == mt5.ORDER_TYPE_SELL and current_price >= pos.sl)
            ):
                logging.info(f"Position {pos.ticket} hit stop loss, should be auto-closed by broker")
                return
            
            # Close if hit take profit (safety check)
            if pos.tp > 0 and (
                (pos.type == mt5.ORDER_TYPE_BUY and current_price >= pos.tp) or
                (pos.type == mt5.ORDER_TYPE_SELL and current_price <= pos.tp)
            ):
                logging.info(f"Position {pos.ticket} hit take profit, should be auto-closed by broker")
                return
            
        except Exception as e:
            logging.error(f"Error checking close conditions for ticket {pos.ticket}: {e}")
    
    def _close_position(self, pos: Any):
        """Close a position (emergency or risk management)"""
        try:
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                logging.error(f"Cannot close ticket {pos.ticket}: no tick data")
                return
            
            # Determine price
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": price,
                "comment": "Risk_Management_Close",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[CLOSE] Position {pos.ticket} closed at {price:.5f} (P&L: {pos.profit:.2f})")
            else:
                logging.error(f"Failed to close position {pos.ticket}: {result.comment}")
                
        except Exception as e:
            logging.error(f"Error closing position {pos.ticket}: {e}")
            
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
                
                # Combined signal logic - only open NEW trades for configured symbols
                open_positions = self.get_open_positions(symbol)
                if open_positions < self.max_positions:
                    if self.use_ml:
                        if ml_signal != 0 and ml_confidence >= self.ml_threshold:
                            self.open_trade(symbol, ml_signal, ml_confidence)
                    else:
                        if smc_signal != 0:
                            self.open_trade(symbol, smc_signal, 0.5)
                
                # Manage ALL open trades (bot-generated and manual)
                self.manage_all_trades()
                
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
        if self.use_ml:
            logging.info(
                f"ML runtime settings: timeframe={self.timeframe} | "
                f"ml_threshold={self.ml_threshold:.2f}"
            )
        
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

    # Always trade symbols discovered from ALL_MODELS folders
    discovered_symbols = MultiSymbolAutoTrader.discover_symbols_from_models()
    symbols_list = discovered_symbols
    if not symbols_list:
        print("No valid model folders found in ../ALL_MODELS")
        print("Train models first: python TRAIN_CRYPTO_MODELS.py")
        exit(1)

    print(f"Using model-folder symbols only: {', '.join(symbols_list)}")
    
    TIMEFRAME = str(trading_cfg.get("timeframe", "1H")).upper()
    RISK_PERCENT = float(trading_cfg.get("risk_percent", 1.0))
    MAX_POSITIONS = int(trading_cfg.get("max_positions", 3))
    USE_ML = bool(trading_cfg.get("use_ml", True))
    ML_THRESHOLD = float(trading_cfg.get("ml_threshold", 0.55))
    CHECK_INTERVAL = int(execution_cfg.get("check_interval", 300))
    
    # Create trader
    trader = MultiSymbolAutoTrader(
        symbols=symbols_list,
        timeframe=TIMEFRAME,
        risk_percent=RISK_PERCENT,
        max_positions=MAX_POSITIONS,
        use_ml=USE_ML,
        ml_threshold=ML_THRESHOLD
    )
    
    # Initialize MT5
    if not trader.initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH):
        print("Failed to initialize MT5")
        exit(1)

    # Keep only symbols available on the connected broker account
    available_symbols, skipped_symbols = trader.filter_symbols_by_mt5_availability()
    if skipped_symbols:
        print(f"Skipping unavailable broker symbols: {', '.join(skipped_symbols)}")
    if not available_symbols:
        print("No model-folder symbols are available on this broker account")
        exit(1)
    print(
        "Startup symbol summary: "
        f"discovered={len(discovered_symbols)} | "
        f"available={len(available_symbols)} | "
        f"skipped={len(skipped_symbols)}"
    )
    print(f"Trading available model symbols: {', '.join(available_symbols)}")
        
    # Load models
    if not trader.load_models():
        print("Failed to load models")
        exit(1)
        
    # Start trading
    trader.run(check_interval=CHECK_INTERVAL)
