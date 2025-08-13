import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from services.binance_service import BinanceService
from services.milvus_service import MilvusService
from services.llm_service import LLMService
from services.base import logging

# Load environment variables
load_dotenv()

# Initialize services
logger = logging.getLogger(__name__)

class TradingAssistant:
    def __init__(self):
        self.embedder = None
        self.binance_service = BinanceService()
        self.milvus_service = MilvusService()
        self.llm_service = LLMService()
        self.scheduler = None
        self._initialization_lock = threading.Lock()
        self._initialized = False
        
    def initialize(self):
        """Initialize all services with proper error handling"""
        with self._initialization_lock:
            if self._initialized:
                return
                
            try:
                logger.info("Initializing Trading Assistant...")
                
                # Initialize embedder
                logger.info("Loading sentence transformer...")
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                
                # Initialize Milvus
                logger.info("Connecting to Milvus...")
                self.milvus_service.connect()
                dim = self.embedder.get_sentence_embedding_dimension()
                self.collection = self.milvus_service.get_or_create_collection(dim)
                
                # Warm up LLM
                logger.info("Warming up LLM...")
                threading.Thread(target=self.llm_service.warm_up_model, daemon=True).start()
                
                # Start scheduler
                self._start_scheduler()
                
                self._initialized = True
                logger.info("Trading Assistant initialized successfully")
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                raise
    
    def _start_scheduler(self):
        """Start the market data collection scheduler"""
        if self.scheduler is not None:
            return
            
        from apscheduler.schedulers.background import BackgroundScheduler
        
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            func=self._market_data_job,
            trigger="interval",
            minutes=15,
            id="market_data_collection",
            replace_existing=True,
            max_instances=1
        )
        self.scheduler.start()
        logger.info("Market data scheduler started")
        
        # Run initial job
        threading.Thread(target=self._market_data_job, daemon=True).start()
    
    def _market_data_job(self):
        """Collect and store market data"""
        try:
            logger.info("Starting market data collection job...")
            
            # Fetch market data
            all_entries = self.binance_service.fetch_all_symbols()
            
            if not all_entries:
                logger.warning("No market data collected")
                return
            
            # Check for new entries
            new_entries = self.milvus_service.check_existing_documents(all_entries)
            
            if new_entries:
                # Generate embeddings
                embeddings = self.embedder.encode(new_entries)
                
                # Store in Milvus
                self.milvus_service.insert_documents(new_entries, embeddings.tolist())
                logger.info(f"Stored {len(new_entries)} new market entries")
            else:
                logger.info("No new market data to store")
                
        except Exception as e:
            logger.error(f"Market data job failed: {e}")
    
    def query(self, question: str) -> dict:
        """Process user query"""
        if not self._initialized:
            self.initialize()
        
        try:
            # Determine if it's a trading-related query
            is_trading_query = self._is_trading_query(question)
            
            # Get context from vector store
            query_embedding = self.embedder.encode([question])[0]
            context_docs = self.milvus_service.search(query_embedding.tolist(), top_k=3)
            context = "\n".join(context_docs)
            
            # Get live market data if relevant
            market_data = None
            if is_trading_query:
                market_data = self._get_live_market_data()
            
            # Generate response
            llm_response = self.llm_service.generate_response(
                question=question,
                context=context,
                market_data=market_data
            )
            
            # Build response
            response = {
                "answer": llm_response["answer"],
                "status": llm_response.get("status", "success"),
                "context": context_docs[:2],  # Include some context
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if market_data:
                response["market_data"] = market_data
            
            if llm_response.get("error"):
                response["error"] = llm_response["error"]
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I encountered an error processing your request. Please try again.",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _is_trading_query(self, question: str) -> bool:
        """Check if question is trading-related"""
        trading_keywords = [
            "bitcoin", "crypto", "stock", "trade", "trading", "market", "price",
            "btc", "eth", "ethereum", "binance", "futures", "spot", "volume",
            "technical", "analysis", "chart", "candlestick", "support", "resistance"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in trading_keywords)
    
    def _get_live_market_data(self) -> dict:
        """Get current market data for major symbols"""
        market_data = {}
        
        for symbol in self.binance_service.symbols:
            try:
                price_data = self.binance_service.get_live_price(symbol)
                if price_data:
                    market_data[symbol] = price_data
            except Exception as e:
                logger.warning(f"Failed to get live price for {symbol}: {e}")
                continue
        
        return market_data

# Initialize global instance
trading_assistant = TradingAssistant()

# Flask app setup
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query_endpoint():
    """Main query endpoint"""
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        
        if not question:
            return jsonify({
                "error": "Missing 'question' parameter",
                "status": "error"
            }), 400
        
        if len(question.strip()) < 3:
            return jsonify({
                "error": "Question too short",
                "status": "error"
            }), 400
        
        # Process query
        response = trading_assistant.query(question)
        
        # Return appropriate HTTP status based on response status
        status_code = 200
        if response.get("status") == "error":
            status_code = 500
        elif response.get("status") == "fallback":
            status_code = 206  # Partial content
        
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route("/health", methods=["GET"])
def health_endpoint():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check Milvus connection
        try:
            trading_assistant.milvus_service.ensure_connection()
            health_status["services"]["milvus"] = "connected"
        except Exception as e:
            health_status["services"]["milvus"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check LLM service
        try:
            if trading_assistant.llm_service._model_warmed:
                health_status["services"]["llm"] = "ready"
            else:
                health_status["services"]["llm"] = "warming_up"
        except Exception as e:
            health_status["services"]["llm"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check Binance API
        try:
            test_price = trading_assistant.binance_service.get_live_price("BTCUSDT")
            if test_price:
                health_status["services"]["binance"] = "connected"
            else:
                health_status["services"]["binance"] = "unavailable"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["binance"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error",
        "timestamp": datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "status": "error",
        "timestamp": datetime.utcnow().isoformat()
    }), 500

# Main execution
if __name__ == "__main__":
    try:
        # Initialize in main thread to avoid issues
        logger.info("Starting Trading Assistant application...")
        
        # Only start scheduler in main process (not in reloader)
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
            trading_assistant.initialize()
        
        # Start Flask app
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=True,
            use_reloader=False,  # Disable reloader to prevent duplicate schedulers
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Cleanup
        if trading_assistant.scheduler and trading_assistant.scheduler.running:
            trading_assistant.scheduler.shutdown(wait=True)
            logger.info("Scheduler shut down")