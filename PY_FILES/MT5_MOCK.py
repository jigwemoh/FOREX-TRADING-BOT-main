#!/usr/bin/env python3
"""
Mock MT5 for testing on non-Windows systems
WARNING: This is for DEVELOPMENT ONLY - NOT for real trading!
"""
from typing import Dict, List, Any, Optional

class MockTick:
    def __init__(self):
        self.ask = 1.10500
        self.bid = 1.10490
        
class MockSymbolInfo:
    def __init__(self):
        self.point = 0.00001
        self.trade_tick_value = 1.0
        self.volume_min = 0.01
        self.volume_max = 100.0
        self.volume_step = 0.01
        
class MockAccountInfo:
    def __init__(self):
        self.balance = 10000.0
        self.equity = 10000.0
        self.margin = 0.0
        self.margin_free = 10000.0
        self.profit = 0.0
        
class MockPosition:
    def __init__(self, symbol: str, type_val: int, volume: float) -> None:
        self.symbol = symbol
        self.type = type_val
        self.type_str = "buy" if type_val == 0 else "sell"
        self.volume = volume
        self.profit = 0.0
        self.sl = 0.0
        self.tp = 0.0
        self.ticket = 12345
        
class MockOrderResult:
    def __init__(self):
        self.retcode = 10009  # TRADE_RETCODE_DONE
        self.order = 12345
        self.comment = "Mock order"

# Constants
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 60
TIMEFRAME_H4 = 240
TIMEFRAME_D1 = 1440

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 2

ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1

TRADE_RETCODE_DONE = 10009

def initialize():
    print("Mock MT5: Initialized (DEVELOPMENT MODE)")
    return True
    
def login(login: int, password: str, server: str) -> bool:
    print(f"Mock MT5: Login {login} to {server} (DEVELOPMENT MODE)")
    return True
    
def shutdown():
    print("Mock MT5: Shutdown (DEVELOPMENT MODE)")
    
def account_info():
    return MockAccountInfo()
    
def symbol_info(symbol: str) -> MockSymbolInfo:
    return MockSymbolInfo()
    
def symbol_info_tick(symbol: str) -> MockTick:
    return MockTick()
    
def copy_rates_from_pos(symbol: str, timeframe: int, start: int, count: int):
    import numpy as np
    import time
    
    # Generate mock data
    current_time = int(time.time())
    times = [current_time - (i * 3600) for i in range(count)][::-1]
    
    data = np.array(
        [(t, 1.105 + np.random.randn()*0.001, 
          1.106 + np.random.randn()*0.001,
          1.104 + np.random.randn()*0.001,
          1.105 + np.random.randn()*0.001,
          1000 + np.random.randint(0, 500))
         for t in times],
        dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
               ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8')]
    )
    return data
    
def positions_get(symbol: Optional[str] = None) -> List[MockPosition]:
    # Return empty list (no positions)
    return []
    
def order_send(request: Dict[str, Any]) -> MockOrderResult:
    print(f"Mock MT5: Order sent - {request.get('type')} {request.get('symbol')} {request.get('volume')} lots")
    return MockOrderResult()
    
def last_error():
    return (0, "No error")

print("=" * 60)
print("WARNING: Using MOCK MT5 module for development")
print("This is NOT connected to real MetaTrader 5")
print("DO NOT use this for live trading!")
print("=" * 60)
