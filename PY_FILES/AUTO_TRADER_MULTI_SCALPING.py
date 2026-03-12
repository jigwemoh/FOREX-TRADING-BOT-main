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
from datetime import datetime, timezone

# ===== SCALPING IMPORTS (NEW) =====
from SCALPING_ENGINE import (
    ScalpingEngine, 
    create_scalping_setup_from_ml,
    MLSetupQualityPredictor,
    AdaptiveParameterManager,
    DynamicExitManager,
    AdvancedRiskManager,
    PerformanceAnalytics
)
from SCALPING_INTEGRATION import ScalpingIntegration, RiskAdjustmentEngine

# ===== PHASE 3 ORDER FLOW IMPORTS (NEW) =====
from ORDER_FLOW_ANALYZER import OrderFlowAnalyzer
from HIGH_FREQUENCY_PROCESSOR import AsyncTickHandler, MultiScaleAnalyzer, LatencyOptimizer
from ORDER_FLOW_SIGNAL_GENERATOR import OrderFlowSignalGenerator, ConfluenceAnalyzer, RegimeAdaptiveWeighter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../CSV_FILES/trading_log.txt', encoding='utf-8'),
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
        max_positions_global: int = 5,
        use_ml: bool = True,
        ml_threshold: float = 0.55
    ):
        self.symbols = symbols or ["EURUSD"]
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.max_positions = max_positions  # Per-symbol
        self.max_positions_global = max_positions_global  # Total across all symbols
        self.use_ml = use_ml
        self.ml_threshold = ml_threshold
        self.models: Dict[str, Dict[str, object]] = {sym: {} for sym in self.symbols}
        self.scalers: Dict[str, Dict[str, object]] = {sym: {} for sym in self.symbols}
        self.model_features: Dict[str, List[str]] = {sym: [] for sym in self.symbols}
        self.symbol_optimal_thresholds: Dict[str, float] = {}  # Per-symbol optimal thresholds
        self._feature_gap_logged: Set[str] = set()
        self._ml_diag_logged_symbols: Set[str] = set()
        self.is_running = False
        
        # Risk Management Thresholds
        self.max_daily_loss = 100.0  # Max loss before stopping trading ($)
        self.max_account_drawdown_percent = 5.0  # Max drawdown % before closing all positions
        self.max_consecutive_losses = 5  # Stop opening new trades after N consecutive losses
        self.consecutive_loss_counter = 0  # Track consecutive losing trades
        
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

        # ===== SCALPING COMPONENTS INITIALIZATION =====
        self.scalping_engines: Dict[str, ScalpingEngine] = {}
        self.scalping_integration = ScalpingIntegration()
        self.ml_quality_predictor = MLSetupQualityPredictor()
        self.adaptive_param_manager = AdaptiveParameterManager()

        # Initialize scalping engines for each symbol
        for symbol in self.symbols:
            self.scalping_engines[symbol] = ScalpingEngine(
                symbol=symbol,
                timeframe="M5",  # Scalping on M5 timeframe
                atr_period=14,
                vwap_lookback=50,
                ema_fast=20,
                ema_slow=50
            )

        # Advanced risk and performance management
        self.advanced_risk_manager = AdvancedRiskManager()
        self.performance_analytics = PerformanceAnalytics()
        self.exit_managers: Dict[int, DynamicExitManager] = {}  # ticket -> exit manager
        self.active_exits: Dict[int, Dict[str, Any]] = {}  # ticket -> exit tracking data

        # ===== PHASE 3 ORDER FLOW COMPONENTS INITIALIZATION =====
        self.order_flow_analyzers: Dict[str, OrderFlowAnalyzer] = {}
        self.async_tick_handlers: Dict[str, AsyncTickHandler] = {}
        self.multi_scale_analyzers: Dict[str, MultiScaleAnalyzer] = {}
        self.order_flow_signal_generators: Dict[str, OrderFlowSignalGenerator] = {}
        self.latency_optimizers: Dict[str, LatencyOptimizer] = {}

        # Initialize Phase 3 components for each symbol
        for symbol in self.symbols:
            # Order flow analyzer for real-time tick analysis
            self.order_flow_analyzers[symbol] = OrderFlowAnalyzer(
                symbol=symbol,
                tick_buffer_size=1000,  # Store last 1000 ticks
                volume_profile_bins=20
            )

            # Async tick handler for high-frequency processing
            self.async_tick_handlers[symbol] = AsyncTickHandler(
                symbol=symbol,
                max_latency_ms=50  # 50ms max latency
            )

            # Multi-scale analyzer for concurrent timeframe analysis
            self.multi_scale_analyzers[symbol] = MultiScaleAnalyzer(symbol=symbol)

            # Order flow signal generator for confluence analysis
            self.order_flow_signal_generators[symbol] = OrderFlowSignalGenerator(symbol=symbol)

            # Latency optimizer for performance monitoring
            self.latency_optimizers[symbol] = LatencyOptimizer(target_latency_ms=50)

        logging.info("[INIT] Phase 3 order flow components initialized with async processing and confluence analysis")

    def run(self, check_interval: int = 300):
        """Main execution method - starts trading threads for all symbols with Phase 3 order flow integration"""
        self.is_running = True
        logger.info(f"[START] Beginning trading execution for {len(self.symbols)} symbols")

        # Start async tick processing for all symbols
        for symbol in self.symbols:
            try:
                # Start async tick handler for real-time order flow processing
                self.async_tick_handlers[symbol].start_processing()
                logger.info(f"[PHASE3] Started async tick processing for {symbol}")
            except Exception as e:
                logger.error(f"[PHASE3] Failed to start async processing for {symbol}: {e}")

        # Create trading threads for each symbol
        threads = []
        for symbol in self.symbols:
            thread = Thread(target=self.run_symbol, args=(symbol, check_interval))
            thread.daemon = True
            threads.append(thread)
            thread.start()
            logger.info(f"[THREAD] Started trading thread for {symbol}")

        # Keep main thread alive and monitor
        try:
            while self.is_running:
                time.sleep(10)  # Check every 10 seconds

                # Monitor Phase 3 components health
                for symbol in self.symbols:
                    try:
                        # Check latency performance
                        latency_stats = self.latency_optimizers[symbol].get_latency_stats()
                        if latency_stats.get('avg_latency_ms', 0) > 100:
                            logger.warning(f"[PHASE3] High latency detected for {symbol}: {latency_stats}")

                        # Check order flow buffer health
                        buffer_size = len(self.order_flow_analyzers[symbol].tick_buffer)
                        if buffer_size < 100:
                            logger.warning(f"[PHASE3] Low tick buffer size for {symbol}: {buffer_size}")

                    except Exception as e:
                        logger.error(f"[PHASE3] Error monitoring {symbol}: {e}")

        except KeyboardInterrupt:
            logger.info("[STOP] Received shutdown signal")
            self.is_running = False

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)

        # Shutdown Phase 3 components
        for symbol in self.symbols:
            try:
                self.async_tick_handlers[symbol].stop_processing()
                logger.info(f"[PHASE3] Stopped async processing for {symbol}")
            except Exception as e:
                logger.error(f"[PHASE3] Error stopping {symbol}: {e}")

        logger.info("[STOP] Trading execution completed")

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
    def discover_symbols_from_models(models_root: Path = None) -> List[str]:
        """Discover tradable symbols from model folders only."""
        if models_root is None:
            # Resolve relative to script location, not current working directory
            script_dir = Path(__file__).parent
            models_root = script_dir.parent / "ALL_MODELS"
        
        if not models_root.exists():
            logging.warning(f"Models root not found: {models_root}")
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

        logging.info(f"[DISCOVERY] Found {len(symbols)} symbols with models in {models_root}")
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

        # ===== SCALPING ENGINE INITIALIZATION (NEW) =====
        # Initialize scalping engines for intraday trading
        self.scalping_engines: Dict[str, ScalpingEngine] = {}
        self.scalping_integration = ScalpingIntegration(config_path="../config.json")
        self.risk_adjuster = RiskAdjustmentEngine(initial_risk=0.02)
        
        # ===== ADVANCED SCALPING COMPONENTS (NEW) =====
        self.exit_managers: Dict[str, DynamicExitManager] = {}
        self.advanced_risk_manager = AdvancedRiskManager()
        self.performance_analytics = PerformanceAnalytics()
        self.active_exits: Dict[str, Dict[str, Any]] = {}  # ticket -> exit info
        
        for symbol in self.symbols:
            self.scalping_engines[symbol] = ScalpingEngine(
                symbol=symbol,
                timeframe="M1",
                atr_period=14,
                vwap_lookback=50,
                ema_fast=20,
                ema_slow=50
            )
        
        logging.info("[SCALPING] Engines initialized for all symbols")
        logging.info("[ADVANCED] Exit managers and analytics initialized")
        
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
        
        # Resolve models directory relative to script location
        script_dir = Path(__file__).parent
        models_root = script_dir.parent / "ALL_MODELS"
        
        logging.info(f"[MODELS] Loading from: {models_root}")
        
        loaded_count = 0
        for symbol in self.symbols:
            model_dir = models_root / symbol
            if not model_dir.exists():
                logging.warning(f"[MODELS] No models found for {symbol} at {model_dir}")
                continue
                
            try:
                features_path = model_dir / "features.joblib"
                if features_path.exists():
                    loaded_features = joblib.load(features_path)
                    if isinstance(loaded_features, list):
                        self.model_features[symbol] = [str(feature) for feature in loaded_features]
                        logging.info(
                            f"[MODELS] Loaded feature metadata: {symbol} ({len(self.model_features[symbol])} features)"
                        )
                    else:
                        logging.warning(f"[MODELS] Invalid features list for {symbol}, using empty list")

                # Load symbol-specific optimal threshold from metadata
                metadata_path = model_dir / "model_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        # Try to get timeframe-specific threshold, fallback to global
                        tf_key = self.timeframe.replace("_", "").upper()  # 1H -> 1H
                        if tf_key in metadata:
                            optimal_thresh = metadata[tf_key].get("optimal_threshold", self.ml_threshold)
                        else:
                            optimal_thresh = metadata.get("optimal_threshold", self.ml_threshold)
                        
                        # Use optimal threshold but cap at 0.65 max to avoid over-fitting to backtest data
                        # (metadata thresholds like 0.75 are unrealistic for live trading)
                        # Then ensure we're at least as strict as global threshold
                        capped_thresh = min(optimal_thresh, 0.65)  # Cap metadata thresholds
                        self.symbol_optimal_thresholds[symbol] = max(capped_thresh, self.ml_threshold)  # But respect global min
                        logging.info(
                            f"[MODELS] [{symbol}] Using optimal threshold: {self.symbol_optimal_thresholds[symbol]:.2f} "
                            f"(metadata: {optimal_thresh:.2f}, capped: {capped_thresh:.2f}, global: {self.ml_threshold:.2f})"
                        )
                else:
                    self.symbol_optimal_thresholds[symbol] = self.ml_threshold
                    logging.warning(f"[MODELS] [{symbol}] No metadata found, using global threshold: {self.ml_threshold:.2f}")

                # Load models for different timeframes (intraday and daily/higher)
                timeframes = ["T_5M", "T_10M", "T_15M", "T_20M", "T_30M", "T_1H", "T_4H", "T_1D", "T_1W", "T_1M"]
                for tf in timeframes:
                    model_path = model_dir / f"{tf}.joblib"
                    scaler_path = model_dir / f"{tf}_scaler.joblib"
                    
                    if model_path.exists() and scaler_path.exists():
                        try:
                            self.models[symbol][tf] = joblib.load(model_path)
                            self.scalers[symbol][tf] = joblib.load(scaler_path)
                            loaded_count += 1
                            logging.info(f"[MODELS] Loaded: {symbol}/{tf}")
                        except Exception as e:
                            logging.error(f"[MODELS] Error loading {symbol}/{tf}: {e}")
                        
            except Exception as e:
                logging.error(f"[MODELS] Error loading models for {symbol}: {e}")
        
        if loaded_count == 0 and self.use_ml:
            logging.error(f"[MODELS] No models loaded ({loaded_count} files), trading without ML")
            self.use_ml = False
            return False
        
        logging.info(f"[MODELS] Successfully loaded {loaded_count} model files for {len(self.models)} symbols")
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

                prob_sell = float(probabilities[0])  # Probability of price going down
                prob_buy = float(probabilities[1])   # Probability of price going up
                confidence = max(prob_buy, prob_sell)

                # Use symbol-specific optimal threshold
                base_threshold = self.symbol_optimal_thresholds.get(symbol, self.ml_threshold)
                
                # Use lower threshold for BUY signals to balance the bias towards SELL predictions
                buy_threshold = base_threshold * 0.8  # 20% lower threshold for BUY
                sell_threshold = base_threshold
                
                # Log signal details periodically
                if diag_key not in self._ml_diag_logged_symbols or np.random.random() < 0.05:  # Log 5% of signals
                    logging.info(
                        f"[ML SIGNAL] {symbol} | BUY={prob_buy:.3f} SELL={prob_sell:.3f} | "
                        f"buy_thresh={buy_threshold:.2f} sell_thresh={sell_threshold:.2f} | "
                        f"signal={'BUY' if prob_buy >= buy_threshold and prob_buy > prob_sell else 'SELL' if prob_sell >= sell_threshold and prob_sell > prob_buy else 'HOLD'}"
                    )

                if prob_buy >= buy_threshold and prob_buy > prob_sell:
                    return 1, confidence
                if prob_sell >= sell_threshold and prob_sell > prob_buy:
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
    
    def _close_all_positions(self):
        """Emergency close all open positions (critical loss management)"""
        try:
            all_positions = self.get_all_open_positions()
            logging.warning(f"[EMERGENCY] Closing all {len(all_positions)} open positions")
            
            for pos in all_positions:
                try:
                    self._close_position(pos)
                    time.sleep(0.5)  # Brief delay between closes
                except Exception as e:
                    logging.error(f"Error closing position {pos.ticket}: {e}")
                    
        except Exception as e:
            logging.error(f"Error in emergency close all: {e}")
            
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
            
            # Risk calculation: risk_percent of account balance
            # Example: 1% of $26,799 = $267.99 risk per trade
            risk_money = account.balance * (self.risk_percent / 100)
            
            # Standard stop loss: 50 pips (can be adjusted per symbol)
            # Lot size = Risk Money / (Stop Loss in pips * Tick Value)
            stop_loss_pips = 50
            lot_size = risk_money / (stop_loss_pips * tick_value)
            
            # Enforce symbol's volume constraints
            lot_step = symbol_info.volume_step
            lot_min = symbol_info.volume_min
            lot_max = symbol_info.volume_max
            
            # Clamp and round to symbol's step
            lot_size = max(lot_min, min(lot_size, lot_max))
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Log lot calculation for transparency
            logging.info(
                f"[{symbol}] Lot calculation: balance=${account.balance:.2f} | "
                f"risk={self.risk_percent}% (${risk_money:.2f}) | SL={stop_loss_pips}pips | "
                f"tick_value=${tick_value:.6f} | calculated={risk_money / (stop_loss_pips * tick_value):.4f} | "
                f"final={lot_size:.6f} (min={lot_min}, max={lot_max}, step={lot_step})"
            )
            
            # Get market data for entry
            df = self.get_market_data(symbol, 100)
            if df is None:
                return
                
            current_price = df['Close'].iloc[-1]
            
            # Calculate SL and TP with proper point conversion
            # Each pip = 10 * point (for most currency pairs)
            point_value = symbol_info.point
            pip_distance = stop_loss_pips * 10 * point_value  # 50 pips in price units
            
            # Ensure minimum distance based on symbol's ask/bid spread
            ask_bid_spread = symbol_info.ask - symbol_info.bid if hasattr(symbol_info, 'ask') else 0
            min_distance = max(pip_distance, ask_bid_spread * 2)  # At least 2x spread
            
            # Create order
            if signal == 1:
                order_type = mt5.ORDER_TYPE_BUY
                order_comment = f"ML_BUY_{confidence:.2f}"
                sl = current_price - min_distance
                tp = current_price + (min_distance * 2)
            else:
                order_type = mt5.ORDER_TYPE_SELL
                order_comment = f"ML_SELL_{confidence:.2f}"
                sl = current_price + min_distance
                tp = current_price - (min_distance * 2)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": current_price,
                "sl": sl,
                "tp": tp,
                "comment": order_comment,
                "type_filling": mt5.ORDER_FILLING_FOK,
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
                # ===== CRITICAL LOSS MANAGEMENT CHECK =====
                account_info = mt5.account_info()
                if account_info:
                    # Check daily loss limit
                    if account_info.profit < -self.max_daily_loss:
                        logging.error(
                            f"[CRITICAL] Daily loss limit exceeded: ${account_info.profit:.2f} < -${self.max_daily_loss}. "
                            f"STOPPING ALL TRADING"
                        )
                        self.is_running = False
                        self._close_all_positions()
                        break
                    
                    # Check account drawdown
                    drawdown_percent = (account_info.profit / account_info.balance) * 100 if account_info.balance > 0 else 0
                    if drawdown_percent < -self.max_account_drawdown_percent:
                        logging.warning(
                            f"[RISK] Account drawdown: {drawdown_percent:.2f}% (limit: {-self.max_account_drawdown_percent}%). "
                            f"Pausing new trades, managing existing positions only"
                        )
                        # Don't open new trades, but continue managing existing ones
                        time.sleep(check_interval)
                        continue
                    
                    # Check consecutive losses (count trades in last hour)
                    all_positions = self.get_all_open_positions()
                    if len(all_positions) > 0:
                        # Count positions with losses
                        loss_count = sum(1 for pos in all_positions if pos.get('profit', 0) < 0)
                        if loss_count >= self.max_consecutive_losses:
                            logging.warning(
                                f"[RISK] Too many losing positions ({loss_count}/{self.max_consecutive_losses}). "
                                f"Pausing new trades"
                            )
                            time.sleep(check_interval)
                            continue
                
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
                
                # Get symbol-specific threshold for logging
                effective_threshold = self.symbol_optimal_thresholds.get(symbol, self.ml_threshold)
                
                logging.info(
                    f"[{symbol}] ML: {ml_signal:+d} (conf={ml_confidence:.3f}, thresh={effective_threshold:.2f}) | "
                    f"SMC: {smc_signal:+d}"
                )
                
                # Combined signal logic - only open NEW trades for configured symbols
                open_positions = self.get_open_positions(symbol)
                total_positions = len(self.get_all_open_positions())
                
                if open_positions >= self.max_positions:
                    logging.info(f"[{symbol}] [BLOCK] Max positions reached ({open_positions}/{self.max_positions})")
                elif total_positions >= self.max_positions_global:
                    logging.info(f"[{symbol}] [BLOCK] Global position limit reached ({total_positions}/{self.max_positions_global})")
                elif self.use_ml:
                    if ml_signal != 0:
                        if ml_confidence >= effective_threshold:
                            logging.info(f"[{symbol}] [PASS] SIGNAL PASSED (conf={ml_confidence:.3f} >= {effective_threshold:.2f}) - Opening trade")
                            self.open_trade(symbol, ml_signal, ml_confidence)
                        else:
                            logging.info(
                                f"[{symbol}] [BLOCK] CONFIDENCE TOO LOW: {ml_confidence:.3f} < {effective_threshold:.2f}"
                            )
                    else:
                        logging.info(f"[{symbol}] [HOLD] No ML signal generated (prob_up and prob_down both below threshold)")
                else:
                    if smc_signal != 0:
                        logging.info(f"[{symbol}] [PASS] SMC SIGNAL - Opening trade")
                        self.open_trade(symbol, smc_signal, 0.5)
                    else:
                        logging.info(f"[{symbol}] [HOLD] No SMC signal")

                # Manage ALL open trades (bot-generated and manual)
                self.manage_all_trades()
                
                # Wait for next check
                time.sleep(check_interval)
                
            except Exception as e:
                logging.error(f"[{symbol}] Error in main loop: {e}")
                time.sleep(60)
    
    
    def manage_all_trades(self):
        """
        Advanced trade management with dynamic exits and performance tracking
        """
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None:
                return
            
            current_time = datetime.now(timezone.utc)
            
            for position in positions:
                ticket = position.ticket
                symbol = position.symbol
                direction = 1 if position.type == mt5.ORDER_TYPE_BUY else -1
                entry_price = position.price_open
                current_price = position.price_current
                lot_size = position.volume
                
                # Get current ATR for dynamic stops
                df_current = self._get_market_data_timeframe(symbol, 20, "M5")
                if df_current is not None:
                    current_atr = self._calculate_atr(df_current, 14)
                else:
                    current_atr = 0.0001  # fallback
                
                # Check if this is a scalping trade with exit management
                if ticket in self.exit_managers:
                    exit_info = self.active_exits[ticket]
                    exit_manager = self.exit_managers[ticket]

                    # Update bars held
                    exit_info['bars_held'] += 1

                    # Get current market regime for adaptive exits
                    current_regime = self.scalping_engines[symbol].detect_market_regime(
                        df_current, current_atr, self._get_current_spread(symbol)
                    ).name if df_current is not None else 'NORMAL_VOLATILITY'

                    # Get adaptive exit parameters
                    adaptive_exit_params = self.adaptive_param_manager.get_regime_based_parameters(current_regime)

                    # Apply adaptive trailing stop activation
                    adaptive_trailing_activation = adaptive_exit_params.get('trailing_stop_activation', 1.0)
                    profit_pips = (current_price - entry_price) * direction * 10000  # Approximate pips

                    # Calculate dynamic stop with adaptive parameters
                    dynamic_stop = exit_manager.calculate_dynamic_stop(
                        current_price=current_price,
                        entry_price=entry_price,
                        direction=direction,
                        atr_value=current_atr,
                        bars_held=exit_info['bars_held'],
                        trailing_activation=adaptive_trailing_activation
                    )

                    # Check for scale-out opportunities with adaptive levels
                    adaptive_scale_levels = adaptive_exit_params.get('scale_out_levels', [1.0, 2.0, 3.0])
                    adaptive_scale_sizes = adaptive_exit_params.get('scale_out_sizes', [0.3, 0.3, 0.4])

                    # Update exit manager with adaptive scale levels
                    exit_manager.scale_out_levels = adaptive_scale_levels
                    exit_manager.scale_out_sizes = adaptive_scale_sizes

                    should_scale, scale_size = exit_manager.should_scale_out(
                        current_price=current_price,
                        entry_price=entry_price,
                        direction=direction,
                        position_size=lot_size
                    )

                    if should_scale:
                        # Partial close for profit
                        close_result = self._partial_close_position(ticket, scale_size)
                        if close_result:
                            logging.info(f"[{symbol}] [SCALE OUT] Closed {scale_size:.2f} lots at {current_price:.5f} (adaptive)")

                            # Update sector exposure
                            self.advanced_risk_manager.update_sector_exposure(symbol, scale_size, 'remove')

                    # Check time-based exit with adaptive limits
                    adaptive_max_bars = adaptive_exit_params.get('max_bars_held', {}).get('M5', 8)
                    should_time_exit = exit_manager.should_time_exit(
                        bars_held=exit_info['bars_held'],
                        volatility_regime=current_regime,
                        max_bars_override=adaptive_max_bars
                    )

                    if should_time_exit:
                        # Close entire position due to time limit
                        close_result = self._close_position(ticket, f"Time exit after {exit_info['bars_held']} bars (adaptive)")
                        if close_result:
                            self._record_closed_trade(exit_info, current_price, current_time, "TIME_EXIT")
                            logging.info(f"[{symbol}] [TIME EXIT] Closed after {exit_info['bars_held']} bars (adaptive limit: {adaptive_max_bars})")

                    # Update stop loss if improved
                    current_sl = position.sl
                    if dynamic_stop != current_sl and dynamic_stop != 0:
                        # Only tighten stops, don't loosen them
                        if direction == 1 and dynamic_stop > current_sl:  # Long position
                            self._modify_stop_loss(ticket, dynamic_stop)
                        elif direction == -1 and (dynamic_stop < current_sl or current_sl == 0):  # Short position
                            self._modify_stop_loss(ticket, dynamic_stop)
                
                # Log position status
                pnl = position.profit
                logging.info(
                    f"[MANAGE] {symbol} {direction} | Ticket: {ticket} | "
                    f"Entry: {entry_price:.5f} | Current: {current_price:.5f} | "
                    f"P&L: {pnl:+.2f} | Lot: {lot_size:.2f}"
                )
        
        except Exception as e:
            logging.error(f"[MANAGE] Error in trade management: {e}")

    # ===== MISSING HELPER METHODS =====
    def _record_closed_trade(self, exit_info: Dict[str, Any], close_price: float,
                           close_time: datetime, exit_reason: str):
        """Record closed trade for performance analytics and adaptive learning"""

        try:
            symbol = exit_info['symbol']
            entry_price = exit_info['entry_price']
            direction = exit_info['direction']
            setup_type = exit_info.get('setup_type', 'UNKNOWN')
            bars_held = exit_info.get('bars_held', 0)

            # Calculate P&L
            if direction == 1:  # Long
                pnl_pips = (close_price - entry_price) * 10000
            else:  # Short
                pnl_pips = (entry_price - close_price) * 10000

            is_win = pnl_pips > 0

            # Record in performance analytics
            trade_result = {
                'symbol': symbol,
                'setup_type': setup_type,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl_pips': pnl_pips,
                'bars_held': bars_held,
                'exit_reason': exit_reason,
                'timestamp': close_time.isoformat(),
                'is_win': is_win
            }

            self.performance_analytics.record_trade(trade_result)

            # Record for adaptive parameter learning
            self.adaptive_param_manager.record_performance(symbol, trade_result)

            # Record setup outcome for ML quality predictor
            if hasattr(self, 'ml_quality_predictor') and exit_info.get('setup_features'):
                outcome = 1 if is_win else 0
                self.ml_quality_predictor.record_setup_outcome(exit_info['setup_features'], outcome)

            # Clean up exit tracking
            ticket = None
            for t, info in self.active_exits.items():
                if info == exit_info:
                    ticket = t
                    break

            if ticket:
                if ticket in self.exit_managers:
                    del self.exit_managers[ticket]
                if ticket in self.active_exits:
                    del self.active_exits[ticket]

            logging.info(f"[{symbol}] [TRADE CLOSED] {setup_type} | P&L: {pnl_pips:+.1f}p | "
                        f"Bars: {bars_held} | Reason: {exit_reason}")

        except Exception as e:
            logging.error(f"Error recording closed trade: {e}")

    def _partial_close_position(self, ticket: int, close_volume: float) -> bool:
        """Partially close a position"""
        try:
            # Get position details
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            position = position[0]
            symbol = position.symbol
            direction = position.type
            lot_size = position.volume

            if close_volume >= lot_size:
                # Full close instead
                return self._close_position(ticket, "Scale out to full close")

            # Calculate close price (use current bid/ask)
            if direction == mt5.ORDER_TYPE_BUY:
                # Closing long position - use bid
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    close_price = tick.bid
                else:
                    return False
            else:
                # Closing short position - use ask
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    close_price = tick.ask
                else:
                    return False

            # Execute partial close
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if direction == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": close_price,
                "deviation": 10,
                "magic": 123456,
                "comment": "Partial close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{symbol}] Partial close successful: {close_volume} lots at {close_price}")
                return True
            else:
                logging.error(f"[{symbol}] Partial close failed: {result}")
                return False

        except Exception as e:
            logging.error(f"Error in partial close: {e}")
            return False

    def _close_position(self, ticket: int, comment: str = "") -> bool:
        """Close a position completely"""
        try:
            # Get position details
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            position = position[0]
            symbol = position.symbol
            direction = position.type
            lot_size = position.volume

            # Calculate close price
            if direction == mt5.ORDER_TYPE_BUY:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    close_price = tick.bid
                else:
                    return False
            else:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    close_price = tick.ask
                else:
                    return False

            # Execute close
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL if direction == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": close_price,
                "deviation": 10,
                "magic": 123456,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{symbol}] Position closed: {lot_size} lots at {close_price} ({comment})")
                return True
            else:
                logging.error(f"[{symbol}] Close failed: {result}")
                return False

        except Exception as e:
            logging.error(f"Error closing position: {e}")
            return False

    def _modify_stop_loss(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss for a position"""
        try:
            # Get position details
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            position = position[0]
            symbol = position.symbol

            # Modify SL
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "sl": new_sl,
                "tp": position.tp,  # Keep existing TP
                "position": ticket,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{symbol}] Stop loss modified to {new_sl}")
                return True
            else:
                logging.error(f"[{symbol}] SL modification failed: {result}")
                return False

        except Exception as e:
            logging.error(f"Error modifying stop loss: {e}")
            return False

    # ===== UTILITY METHODS =====
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR value"""
        if len(df) < period + 1:
            return 0.0001

        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return max(atr, 0.0001)
        except:
            return 0.0001

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_current_spread(self, symbol: str) -> float:
        """Get current spread in pips"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                spread_points = tick.ask - tick.bid
                # Convert to pips (assuming 4 decimal places for forex)
                return spread_points * 10000
            return 1.5  # Default spread
        except:
            return 1.5

    def _get_market_data_timeframe(self, symbol: str, bars: int = 100, timeframe: str = "M5") -> Optional[pd.DataFrame]:
        """Get market data for specific timeframe"""
        try:
            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }

            tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

            if rates is None:
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            return df

        except Exception as e:
            logging.error(f"Error getting {timeframe} data for {symbol}: {e}")
            return None

    def get_balance(self) -> float:
        """Get account balance"""
        try:
            account_info = mt5.account_info()
            return account_info.balance if account_info else 10000.0
        except:
            return 10000.0

    def _create_order_flow_scalp_setup(self, symbol: str, order_flow_signal, confluence_score: float,
                                      current_price: float = None, df: pd.DataFrame = None):
        """Create a scalping setup from order flow signals"""
        try:
            if not order_flow_signal or not current_price:
                return None

            # Get scalping engine for the symbol
            scalping_engine = self.scalping_engines.get(symbol)
            if not scalping_engine:
                return None

            # Create a micro-scalping setup based on order flow
            direction = 1 if order_flow_signal.direction == 'bullish' else -1

            # Use tighter stops for order flow scalps (microstructure-based)
            stop_loss_pips = 15  # Tighter than regular scalps
            target_pips = 30     # Smaller target

            # Calculate entry price with micro-adjustment based on order flow
            entry_offset = order_flow_signal.strength * 0.0001  # Micro adjustment
            entry_price = current_price + (entry_offset * direction)

            # Create setup object (similar to regular scalping setup)
            class OrderFlowScalpSetup:
                def __init__(self, setup_type, direction, entry_price, stop_loss_pips, target_pips,
                           volatility_regime, confluence_score):
                    self.setup_type = setup_type
                    self.direction = direction
                    self.entry_price = entry_price
                    self.stop_loss_pips = stop_loss_pips
                    self.target_pips = target_pips
                    self.volatility_regime = volatility_regime
                    self.confluence_score = confluence_score

            # Determine setup type based on order flow signal
            setup_type_map = {
                'volume_imbalance': 'OF_Volume_Imbalance',
                'order_book_pressure': 'OF_Order_Book_Pressure',
                'tick_flow': 'OF_Tick_Flow',
                'volume_profile': 'OF_Volume_Profile',
                'momentum_divergence': 'OF_Momentum_Divergence',
                'regime_shift': 'OF_Regime_Shift',
                'liquidity_void': 'OF_Liquidity_Void',
                'aggression_imbalance': 'OF_Aggression_Imbalance'
            }

            setup_type = setup_type_map.get(order_flow_signal.signal_type, 'OF_Generic')

            # Detect current volatility regime
            if df is not None and len(df) >= 10:
                atr_value = self._calculate_atr(df, 14)
                spread_pips = self._get_current_spread(symbol)
                volatility_regime = scalping_engine.detect_market_regime(df, atr_value, spread_pips)
            else:
                # Default regime
                class DefaultRegime:
                    name = 'NORMAL_VOLATILITY'
                volatility_regime = DefaultRegime()

            # Calculate confidence based on confluence and signal strength
            base_confidence = order_flow_signal.strength * 0.8  # 80% weight on signal strength
            confluence_boost = confluence_score * 0.2           # 20% weight on confluence
            final_confidence = min(1.0, base_confidence + confluence_boost)

            setup = OrderFlowScalpSetup(
                setup_type=setup_type,
                direction=direction,
                entry_price=entry_price,
                stop_loss_pips=stop_loss_pips,
                target_pips=target_pips,
                volatility_regime=volatility_regime,
                confluence_score=confluence_score
            )

            return setup, final_confidence

        except Exception as e:
            logging.error(f"Error creating order flow scalp setup for {symbol}: {e}")
            return None


# ===== MAIN EXECUTION =====
    
    def _partial_close_position(self, ticket: int, close_volume: float) -> bool:
        """Partially close a position"""
        try:
            # Get position details
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            position = position[0]
            symbol = position.symbol
            direction = position.type
            lot_size = position.volume
            
            if close_volume >= lot_size:
                # Full close instead
                return self._close_position(ticket, "Scale out to full close")
            
            # Calculate close price (use current bid/ask)
            if direction == mt5.ORDER_TYPE_BUY:
                # Closing long position - use bid
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    close_price = tick.bid
                else:
                    return False
            else:
                # Closing short position - use ask
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    close_price = tick.ask
                else:
                    return False
            
            # Send partial close order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if direction == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": close_price,
                "deviation": 10,
                "magic": 0,
                "comment": "Partial Close",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logging.error(f"[PARTIAL CLOSE] Error closing {close_volume} lots of ticket {ticket}: {e}")
            return False
    
    def _close_position(self, ticket: int, reason: str = "") -> bool:
        """Close a position completely"""
        try:
            # Get position details
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            position = position[0]
            symbol = position.symbol
            direction = position.type
            lot_size = position.volume
            
            # Get close price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False
            
            close_price = tick.bid if direction == mt5.ORDER_TYPE_BUY else tick.ask
            
            # Send close order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL if direction == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": close_price,
                "deviation": 10,
                "magic": 0,
                "comment": f"Close: {reason}",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            
            if success:
                # Clean up exit management
                if ticket in self.exit_managers:
                    del self.exit_managers[ticket]
                if ticket in self.active_exits:
                    exit_info = self.active_exits[ticket]
                    # Update sector exposure
                    self.advanced_risk_manager.update_sector_exposure(
                        exit_info['symbol'], exit_info.get('original_size', lot_size), 'remove'
                    )
                    del self.active_exits[ticket]
            
            return success
            
        except Exception as e:
            logging.error(f"[CLOSE] Error closing ticket {ticket}: {e}")
            return False
    
    def _modify_stop_loss(self, ticket: int, new_sl: float):
        """Modify stop loss for a position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": 0.0,  # Keep take profit unchanged
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[MODIFY] Updated SL for ticket {ticket} to {new_sl:.5f}")
            else:
                logging.warning(f"[MODIFY] Failed to update SL for ticket {ticket}")
                
        except Exception as e:
            logging.error(f"[MODIFY] Error modifying SL for ticket {ticket}: {e}")
    
    def _record_closed_trade(self, exit_info: Dict[str, Any], close_price: float, close_time: datetime, exit_reason: str):
        """Record completed trade for performance analytics"""
        try:
            entry_price = exit_info['entry_price']
            direction = exit_info['direction']
            symbol = exit_info['symbol']
            
            # Calculate P&L
            if direction == 1:  # Long
                pnl_pips = (close_price - entry_price) / 10**4 * 10000
            else:  # Short
                pnl_pips = (entry_price - close_price) / 10**4 * 10000
            
            # Convert to USD (approximate)
            tick_value = 1.0 if symbol.endswith('USD') else 0.01  # Rough approximation
            pnl_usd = pnl_pips * tick_value
            
            # Record in analytics
            self.performance_analytics.record_trade(
                symbol=symbol,
                setup_type=exit_info['setup_type'],
                direction=direction,
                entry_time=exit_info['entry_time'],
                exit_time=close_time,
                pnl=pnl_usd,
                bars_held=exit_info['bars_held'],
                volatility_regime=exit_info['volatility_regime'],
                session=self.scalping_integration.get_strategy_for_time()[2]['session']
            )
            
            logging.info(
                f"[TRADE CLOSED] {symbol} | Setup: {exit_info['setup_type']} | "
                f"P&L: {pnl_usd:+.2f} | Bars: {exit_info['bars_held']} | Reason: {exit_reason}"
            )
            
        except Exception as e:
            logging.error(f"[RECORD] Error recording closed trade: {e}")
    
    def get_all_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions with detailed information"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'swap': pos.swap,
                    'commission': getattr(pos, 'commission', 0.0)
                })
            
            return position_list
            
        except Exception as e:
            logging.error(f"[POSITIONS] Error getting open positions: {e}")
            return []
    
    def get_balance(self) -> float:
        """Get current account balance"""
        try:
            account_info = mt5.account_info()
            return account_info.balance if account_info else 0.0
        except Exception as e:
            logging.error(f"[BALANCE] Error getting balance: {e}")
            return 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if df is None or len(df) < period + 1:
            return 0.0001  # Default ATR
        
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr = pd.concat([
                high - low,
                (high - close).abs(),
                (low - close).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean().iloc[-1]
            return atr if not pd.isna(atr) else 0.0001
            
        except Exception as e:
            logging.warning(f"[ATR] Error calculating ATR: {e}")
            return 0.0001
    
    def _get_current_spread(self, symbol: str) -> float:
        """Get current bid-ask spread in pips"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                spread_pips = (tick.ask - tick.bid) / 10**4 * 10000
                return spread_pips
            return 1.5  # Default spread
        except Exception as e:
            logging.warning(f"[SPREAD] Error getting spread for {symbol}: {e}")
            return 1.5
    
    def _get_market_data_timeframe(self, symbol: str, bars: int = 100, timeframe: str = "M5") -> Optional[pd.DataFrame]:
        """Get market data for specific timeframe"""
        try:
            # Timeframe mapping
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M10": mt5.TIMEFRAME_M10,
                "M15": mt5.TIMEFRAME_M15,
                "M20": mt5.TIMEFRAME_M20,
                "M30": mt5.TIMEFRAME_M30,
                "1H": mt5.TIMEFRAME_H1,
                "2H": mt5.TIMEFRAME_H2,
                "3H": mt5.TIMEFRAME_H3,
                "4H": mt5.TIMEFRAME_H4,
                "H4": mt5.TIMEFRAME_H4,
                "1D": mt5.TIMEFRAME_D1,
                "D1": mt5.TIMEFRAME_D1,
            }
            
            mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
            if rates is None:
                logging.warning(f"No data for {symbol} on {timeframe}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                'tick_volume': 'volume'
            })
            
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logging.warning(f"Error fetching {timeframe} data for {symbol}: {e}")
            return None
                
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
    
    # ===== SCALPING HELPER METHODS (NEW) =====
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for stop loss sizing
        
        Args:
            df: OHLC dataframe with columns: high, low, close
            period: ATR period (default 14)
            
        Returns:
            ATR value in price units (same scale as price data)
        """
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return float(atr.iloc[-1]) if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0.0001
        except Exception as e:
            logging.warning(f"Error calculating ATR: {e}")
            return 0.0001

    def _get_current_spread(self, symbol: str) -> float:
        """
        Get current bid-ask spread in pips
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread in pips (4 decimal places = 1 pip)
        """
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick and hasattr(tick, 'ask') and hasattr(tick, 'bid'):
                spread_price = tick.ask - tick.bid
                # Convert to pips (4 decimal places for most pairs)
                spread_pips = spread_price * 10**4
                return max(float(spread_pips), 0.1)
        except Exception as e:
            logging.warning(f"Error getting spread for {symbol}: {e}")
        
        # Fallback to default spreads for major pairs
        default_spreads = {
            "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5,
            "AUDUSD": 1.5, "NZDUSD": 2.5, "USDCAD": 1.8,
            "EURGBP": 1.8, "EURJPY": 2.0
        }
        return float(default_spreads.get(symbol, 2.0))

    def _get_market_data_timeframe(
        self,
        symbol: str,
        bars: int = 100,
        timeframe: str = "M5"
    ) -> Optional[pd.DataFrame]:
        """
        Get market data for specific timeframe (M1, M5, 1H, etc)
        Similar to get_market_data but allows custom timeframe
        
        Args:
            symbol: Trading symbol
            bars: Number of bars to fetch
            timeframe: Timeframe string ("M1", "M5", "1H", "4H", etc)
            
        Returns:
            DataFrame with OHLC data or None if error
        """
        try:
            # Map timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "5M": mt5.TIMEFRAME_M5,
                "M5": mt5.TIMEFRAME_M5,
                "10M": mt5.TIMEFRAME_M10,
                "M10": mt5.TIMEFRAME_M10,
                "15M": mt5.TIMEFRAME_M15,
                "M15": mt5.TIMEFRAME_M15,
                "30M": mt5.TIMEFRAME_M30,
                "M30": mt5.TIMEFRAME_M30,
                "1H": mt5.TIMEFRAME_H1,
                "H1": mt5.TIMEFRAME_H1,
                "4H": mt5.TIMEFRAME_H4,
                "H4": mt5.TIMEFRAME_H4,
                "1D": mt5.TIMEFRAME_D1,
                "D1": mt5.TIMEFRAME_D1,
            }
            
            mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
            if rates is None:
                logging.warning(f"No data for {symbol} on {timeframe}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                'tick_volume': 'volume'
            })
            
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logging.warning(f"Error fetching {timeframe} data for {symbol}: {e}")
            return None


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
        print("=" * 70)
        print("ERROR: Invalid MT5 configuration in ../config.json")
        print("=" * 70)
        print("\nRequired fields:")
        print(f"  ✗ MT5 Login: {MT5_LOGIN or 'MISSING'}")
        print(f"  ✗ MT5 Password: {'MISSING' if not MT5_PASSWORD else '***hidden***'}")
        print(f"  ✗ MT5 Server: {MT5_SERVER or 'MISSING'}")
        print("\nTo configure:")
        print("  python CONFIG_MANAGER.py create")
        print("\nTo use demo account (HFMarketsGlobal-Demo):")
        print("  1. Edit ../config.json")
        print("  2. Set mt5.password to your actual account password")
        print("  3. Ensure mt5.login and mt5.server match your account")
        print("=" * 70)
        exit(1)

    # Always trade symbols discovered from ALL_MODELS folders
    discovered_symbols = MultiSymbolAutoTrader.discover_symbols_from_models()
    
    # Use config symbols if specified, otherwise use discovered symbols
    configured_symbols = trading_cfg.get("symbols", [])
    if configured_symbols:
        # Only trade symbols that are both in config AND have models
        symbols_list = [s for s in configured_symbols if s in discovered_symbols]
        print(f"Using configured symbols: {', '.join(symbols_list)}")
    else:
        # Fallback: trade all discovered symbols
        symbols_list = discovered_symbols
        print(f"Using all discovered symbols: {', '.join(symbols_list)}")
    
    if not symbols_list:
        print("No valid symbols configured or found")
        print(f"Configure symbols in config.json trading.symbols")
        exit(1)
    
    TIMEFRAME = str(trading_cfg.get("timeframe", "1H")).upper()
    RISK_PERCENT = float(trading_cfg.get("risk_percent", 1.0))
    MAX_POSITIONS = int(trading_cfg.get("max_positions", 3))
    MAX_POSITIONS_GLOBAL = int(trading_cfg.get("max_positions_global", 5))
    USE_ML = bool(trading_cfg.get("use_ml", True))
    ML_THRESHOLD = float(trading_cfg.get("ml_threshold", 0.55))
    CHECK_INTERVAL = int(execution_cfg.get("check_interval", 300))
    
    # Create trader
    trader = MultiSymbolAutoTrader(
        symbols=symbols_list,
        timeframe=TIMEFRAME,
        risk_percent=RISK_PERCENT,
        max_positions=MAX_POSITIONS,
        max_positions_global=MAX_POSITIONS_GLOBAL,
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
