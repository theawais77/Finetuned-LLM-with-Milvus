import os
import re
import requests
from typing import Dict, Optional
from .base import CircuitBreaker, retry_on_failure, logging

class LLMService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model_name = os.getenv("LLM_MODEL", "trading-model")
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120)
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self._model_warmed = False
        
    def warm_up_model(self):
        """Warm up the model if not already done"""
        if self._model_warmed:
            return
            
        try:
            self.logger.info("Warming up LLM model...")
            response = self._make_request("ping", max_tokens=2, timeout=30)
            if response:
                self._model_warmed = True
                self.logger.info("Model warmed up successfully")
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")
    
    @retry_on_failure(max_retries=1, delay=2.0)  # Limited retries for LLM
    def _make_request(self, prompt: str, max_tokens: int = 256, timeout: int = 180) -> Optional[Dict]:
        """Make request to LLM with circuit breaker"""
        def make_llm_call():
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "raw": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
                "stop": ["</Answer>", "\n\n---"],
                "keep_alive": "10m"
            }
            
            response = self.session.post(
                self.base_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        
        return self.circuit_breaker.call(make_llm_call)
    
    def generate_response(self, 
                         question: str, 
                         context: str, 
                         market_data: Optional[Dict] = None) -> Dict:
        """Generate response with proper error handling"""
        
        # Ensure model is warmed up
        self.warm_up_model()
        
        # Build prompt
        prompt = self._build_prompt(question, context, market_data)
        
        try:
            # Make LLM request
            response = self._make_request(prompt)
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            # Extract and validate answer
            raw_text = response.get("response", "").strip()
            answer = self._extract_answer(raw_text)
            
            if not answer or len(answer.strip()) < 5:
                raise ValueError("Invalid or empty answer extracted")
            
            return {
                "answer": answer,
                "status": "success",
                "raw_response": raw_text[:1000]  # Truncated for logging
            }
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            
            # Return fallback response
            fallback = self._generate_fallback(question, context, market_data)
            return {
                "answer": fallback,
                "status": "fallback",
                "error": str(e)
            }
    
    def _build_prompt(self, question: str, context: str, market_data: Optional[Dict]) -> str:
        """Build structured prompt for LLM"""
        
        market_section = ""
        if market_data:
            market_section = f"\nLive Market Data:\n{self._format_market_data(market_data)}"
        
        prompt = f"""You are a trading assistant. Use the provided context and market data to answer questions about crypto trading and markets.

Context from knowledge base:
{context or "No relevant context found."}
{market_section}

Question: {question}

Instructions:
- Answer directly and concisely
- Use only the provided data - do not invent facts
- If data is insufficient, say so clearly
- Format your response between <Answer> tags

<Answer>
"""
        return prompt
    
    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for prompt"""
        formatted = []
        for symbol, data in market_data.items():
            if isinstance(data, dict) and "price" in data:
                formatted.append(f"{symbol}: ${data['price']} ({data.get('timestamp', 'N/A')})")
            elif isinstance(data, list):
                formatted.append(f"{symbol}: {len(data)} data points available")
        return "\n".join(formatted[:5])  # Limit to avoid prompt bloat
    
    def _extract_answer(self, raw_response: str) -> str:
        """Extract answer from LLM response with multiple fallback methods"""
        if not raw_response:
            return ""
        
        # Method 1: Extract from <Answer> tags
        answer_match = re.search(r'<Answer>\s*(.*?)\s*(?:</Answer>|$)', 
                               raw_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer:
                return self._clean_answer(answer)
        
        # Method 2: Look for structured response patterns
        lines = raw_response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('#', '```', '---')):
                cleaned = self._clean_answer(line)
                if len(cleaned) > 10:  # Reasonable minimum length
                    return cleaned
        
        # Method 3: Use first substantial paragraph
        paragraphs = [p.strip() for p in raw_response.split('\n\n') if p.strip()]
        for para in paragraphs[:2]:  # Check first 2 paragraphs
            cleaned = self._clean_answer(para)
            if len(cleaned) > 10:
                return cleaned
        
        # Method 4: Return cleaned raw response as last resort
        return self._clean_answer(raw_response)
    
    def _clean_answer(self, text: str) -> str:
        """Clean extracted answer text"""
        if not text:
            return ""
        
        # Remove HTML-like tags
        text = re.sub(r'<[^>]*>', '', text)
        
        # Remove common artifacts
        text = re.sub(r'<\|endoftext\|>', '', text)
        text = re.sub(r'\*\*\*+', '', text)
        text = re.sub(r'---+', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _generate_fallback(self, 
                          question: str, 
                          context: str, 
                          market_data: Optional[Dict]) -> str:
        """Generate fallback response when LLM fails"""
        
        # Check if we have market data to provide basic info
        if market_data and any("price" in str(data) for data in market_data.values()):
            prices = []
            for symbol, data in market_data.items():
                if isinstance(data, dict) and "price" in data:
                    prices.append(f"{symbol}: ${data['price']}")
            
            if prices:
                return f"I'm experiencing technical difficulties, but here's the current market data: {', '.join(prices)}"
        
        # Check if context has relevant info
        if context and len(context.strip()) > 20:
            return "I'm having trouble processing your request fully, but based on available data: " + context[:200] + "..."
        
        # Generic fallback
        return "I'm experiencing technical difficulties and cannot process your request at the moment. Please try again in a few minutes."
