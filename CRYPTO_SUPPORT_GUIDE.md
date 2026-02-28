# Multi-Symbol Trading Configuration

This guide explains how to configure and use the multi-symbol trading bot with both forex and cryptocurrency pairs.

## Supported Pairs

### Forex Pairs
- EURUSD (Euro/USD)
- GBPUSD (British Pound/USD)
- USDJPY (USD/Japanese Yen)
- AUDUSD (Australian Dollar/USD)
- NZDUSD (New Zealand Dollar/USD)
- USDCAD (USD/Canadian Dollar)
- USDCHF (USD/Swiss Franc)

### Cryptocurrency Pairs
- BTCUSD (Bitcoin/USD)
- ETHUSD (Ethereum/USD)
- XRPUSD (XRP/USD)
- LTCUSD (Litecoin/USD)
- ADAUSD (Cardano/USD)
- SOLUSD (Solana/USD)
- DOGEUSD (Dogecoin/USD)

## Configuration

### Step 1: Create Configuration
Run the CONFIG_MANAGER to set up your trading configuration:

```bash
python CONFIG_MANAGER.py create
```

When prompted for symbols, enter comma-separated values:

**Examples:**
- Single symbol: `EURUSD`
- Multiple forex: `EURUSD, GBPUSD, USDJPY`
- Forex + Crypto: `EURUSD, BTCUSD, ETHUSD`
- All crypto: `BTCUSD, ETHUSD, XRPUSD, LTCUSD`

### Step 2: Run the Multi-Symbol Trader

```bash
python AUTO_TRADER_MULTI.py
```

This will:
1. Load your configuration from `config.json`
2. Initialize MT5 connection
3. Load ML models for each configured symbol
4. Start independent trading threads for each symbol
5. Monitor and trade across all symbols simultaneously

## How It Works

### Multi-Symbol Architecture
- **Parallel Trading**: Each symbol runs in its own thread, allowing simultaneous monitoring and trading
- **Independent Signals**: ML and SMC signals are calculated separately for each symbol
- **Shared Account**: All trades execute from the same MT5 account with shared risk parameters
- **Per-Symbol Position Limits**: Max positions limit applies to each symbol individually

### Risk Management
- **Risk Percent**: Applied to total account balance, distributed across symbols
- **Stop Loss & Take Profit**: Calculated in pips, adjusted for each symbol's pip value
- **Max Positions**: Limit per symbol (not total across all symbols)

### Example Configuration
If you have:
- Account Balance: $25,000
- Risk Per Trade: 2%
- Max Positions Per Symbol: 3
- Symbols: EURUSD, BTCUSD, ETHUSD

Then:
- Risk per trade = $500
- Maximum 3 EURUSD positions
- Maximum 3 BTCUSD positions
- Maximum 3 ETHUSD positions
- Total possible: 9 positions

## Upgrading from Single-Symbol Bot

If you were using the original `AUTO_TRADER.py` with single symbol:

1. **Run CONFIG_MANAGER** to update your configuration:
   ```bash
   python CONFIG_MANAGER.py create
   ```

2. **Use the new multi-symbol bot**:
   ```bash
   python AUTO_TRADER_MULTI.py
   ```

3. **Original bot still works** - both `AUTO_TRADER.py` and `AUTO_TRADER_MULTI.py` can coexist

## Model Support

For crypto pairs to work with ML predictions, you need trained models:
- Models should be in: `../ALL_MODELS/{SYMBOL}/`
- Required files: `T_5M.joblib`, `T_5M_scaler.joblib`, etc.

If models are not found for a symbol, the bot will:
1. Log a warning
2. Fall back to SMC signals for that symbol
3. Continue trading using technical analysis only

## Performance Considerations

Running many symbols simultaneously can be resource-intensive:

| Symbols | CPU Usage | Memory | Recommendation |
|---------|-----------|--------|-----------------|
| 1-3 | Low | ~100MB | Recommended |
| 3-7 | Medium | ~200MB | Good balance |
| 7+ | High | ~300MB+ | Only on powerful VPS |

## Troubleshooting

### Symbol Not Found
**Error**: `Terminal: Call failed` for a symbol

**Solution**:
1. Verify the symbol is available on your broker
2. Ensure MT5 terminal has the symbol subscribed
3. Check symbol spelling (must be uppercase)
4. Try a different symbol

### Threads Not Starting
**Error**: Fewer threads than symbols

**Solution**:
1. Check MT5 initialization succeeded
2. Verify symbol availability
3. Restart MT5 terminal and try again

### High CPU Usage
**Solution**:
1. Reduce number of symbols
2. Increase check_interval (e.g., 600 seconds instead of 300)
3. Upgrade VPS resources

## Logs and Monitoring

All trading activity is logged to:
- `../CSV_FILES/trading_log.txt` - Detailed logs
- Console output - Real-time updates

Monitor logs to see:
- Symbol initialization
- Signal generation per symbol
- Trade execution
- Error messages per symbol

## Next Steps

1. Configure your symbols in `CONFIG_MANAGER`
2. Ensure MT5 terminal is running and logged in
3. Run `python AUTO_TRADER_MULTI.py`
4. Monitor the logs for successful trades
5. Adjust risk parameters and symbol list based on results
