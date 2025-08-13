import os
import requests
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import CircuitBreaker, retry_on_failure, logging

class BinanceService:
    def __init__(self):
        self.spot_base = "https://api.binance.com"
        self.futures_base = "https://fapi.binance.com"
        self.symbols = ["BTCUSDT", "ETHUSDT"]
        self.circuit_breaker = CircuitBreaker()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingBot/1.0"})
        self.logger = logging.getLogger(__name__)
        
    @retry_on_failure(max_retries=2, delay=1.0)
    def _get_json(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request with circuit breaker and retry logic"""
        def make_request():
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        return self.circuit_breaker.call(make_request)
    
    def fetch_symbol_data(self, symbol: str) -> List[str]:
        """Fetch comprehensive data for a single symbol"""
        timestamp = datetime.utcnow().isoformat()
        entries = []
        
        try:
            # Define all endpoints
            endpoints = [
                ("stats", f"{self.spot_base}/api/v3/ticker/24hr", {"symbol": symbol}),
                ("depth", f"{self.spot_base}/api/v3/depth", {"symbol": symbol, "limit": 10}),
                ("trades", f"{self.spot_base}/api/v3/trades", {"symbol": symbol, "limit": 5}),
                ("klines", f"{self.spot_base}/api/v3/klines", 
                 {"symbol": symbol, "interval": "1m", "limit": 10}),
                ("oi", f"{self.futures_base}/fapi/v1/openInterest", {"symbol": symbol}),
                ("liquidations", f"{self.futures_base}/fapi/v1/allForceOrders", 
                 {"symbol": symbol, "limit": 5}),
                ("long_short", f"{self.futures_base}/futures/data/topLongShortAccountRatio",
                 {"symbol": symbol, "period": "5m", "limit": 1})
            ]
            
            # Fetch data with controlled parallelism
            results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_name = {
                    executor.submit(self._safe_fetch, url, params): name
                    for name, url, params in endpoints
                }
                
                for future in as_completed(future_to_name, timeout=30):
                    name = future_to_name[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch {name} for {symbol}: {e}")
                        results[name] = {"error": str(e)}
            
            # Process results and create entries
            entries.extend(self._process_spot_data(symbol, timestamp, results))
            entries.extend(self._process_futures_data(symbol, timestamp, results))
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            entries.append(f"[{timestamp}] {symbol} Data fetch failed: {str(e)}")
            
        return entries
    
    def _safe_fetch(self, url: str, params: Dict) -> Dict:
        """Safely fetch data from an endpoint"""
        try:
            return self._get_json(url, params)
        except Exception as e:
            return {"error": str(e)}
    
    def _process_spot_data(self, symbol: str, timestamp: str, results: Dict) -> List[str]:
        """Process spot market data"""
        entries = []
        
        # 24hr stats
        stats = results.get("stats", {})
        if "error" not in stats and stats:
            entries.append(
                f"[{timestamp}] {symbol} Spot: ${stats.get('lastPrice', 'N/A')} "
                f"({stats.get('priceChangePercent', 'N/A')}% 24h, "
                f"Vol: {stats.get('volume', 'N/A')})"
            )
        
        # Order book
        depth = results.get("depth", {})
        if "error" not in depth and depth.get("bids") and depth.get("asks"):
            best_bid = depth["bids"][0][0]
            best_ask = depth["asks"][0][0]
            spread = float(best_ask) - float(best_bid)
            entries.append(
                f"[{timestamp}] {symbol} Book: Bid ${best_bid}, Ask ${best_ask}, "
                f"Spread: ${spread:.4f}"
            )
        
        return entries
    
    def _process_futures_data(self, symbol: str, timestamp: str, results: Dict) -> List[str]:
        """Process futures market data"""
        entries = []
        
        # Open Interest
        oi = results.get("oi", {})
        if "error" not in oi and "openInterest" in oi:
            entries.append(
                f"[{timestamp}] {symbol} Futures OI: {oi['openInterest']} contracts"
            )
        
        # Long/Short Ratio
        long_short = results.get("long_short", [])
        if isinstance(long_short, list) and long_short:
            ls_data = long_short[0]
            ratio = f"{ls_data.get('longAccount', 'N/A')}:{ls_data.get('shortAccount', 'N/A')}"
            entries.append(
                f"[{timestamp}] {symbol} L/S Ratio (5m): {ratio}"
            )
        
        return entries
    
    def fetch_all_symbols(self) -> List[str]:
        """Fetch data for all configured symbols"""
        all_entries = []
        
        for symbol in self.symbols:
            try:
                entries = self.fetch_symbol_data(symbol)
                all_entries.extend(entries)
                self.logger.info(f"Fetched {len(entries)} entries for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {e}")
                continue
                
        return all_entries
    
    def get_live_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for a symbol (used for live queries)"""
        try:
            data = self._get_json(
                f"{self.spot_base}/api/v3/ticker/price", 
                {"symbol": symbol}
            )
            return {
                "symbol": symbol,
                "price": data.get("price"),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get live price for {symbol}: {e}")
            return None