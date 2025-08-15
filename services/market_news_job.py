import os
import logging
import requests
from services.milvus_service import MilvusService
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MarketNewsCollector:
    def __init__(self):
        self.milvus_service = MilvusService()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.news_api_key = os.getenv("NEWS_API_KEY")
        if not self.news_api_key:
            raise ValueError("Missing NEWS_API_KEY environment variable")

    def fetch_crypto_news(self):
        try:
            url = "https://newsapi.org/v2/everything"
            keywords = (
                "cryptocurrency OR bitcoin OR ethereum OR altcoin OR blockchain OR "
                "crypto exchange OR Binance OR Coinbase OR crypto crash OR crypto hack "
                "OR crypto regulation OR DeFi OR NFT"
            )
            params = {
                "q": keywords,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 25,
                "apiKey": self.news_api_key
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

            entries = []
            for art in articles:
                published = art.get("publishedAt", "").strip()
                title = art.get("title", "").strip()
                source = art.get("source", {}).get("name", "")
                desc = art.get("description", "").strip() if art.get("description") else ""
                link = art.get("url", "")

                # Final formatted entry for storage
                entries.append(
                    f"[{published}] NEWS ({source}): {title} — {desc} ({link})"
                )

            logger.info(f"Fetched {len(entries)} crypto news articles")
            return entries

        except Exception as e:
            logger.error(f"Error fetching crypto news: {e}")
            return []

    def run(self):
        logger.info("Starting crypto news collection job...")

        # Get news articles
        news_entries = self.fetch_crypto_news()
        if not news_entries:
            logger.warning("No news fetched this run.")
            return

        # Deduplicate
        new_entries = self.milvus_service.check_existing_documents(news_entries)
        if not new_entries:
            logger.info("No new news to store.")
            return

        # Embed and store
        embeddings = self.embedder.encode(new_entries)
        self.milvus_service.insert_documents(new_entries, embeddings.tolist())
        logger.info(f"Stored {len(new_entries)} new news entries in vector DB")
