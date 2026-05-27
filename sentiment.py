import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def _yahoo_news(ticker: str, max_items: int = 10):
    items = []
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return items

    for item in news[:max_items]:
        title = item.get("title", "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "source": item.get("publisher", "Yahoo Finance"),
            "url": item.get("link", "")
        })
    return items

def _google_news_rss(ticker: str, max_items: int = 10):
    items = []
    query = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception:
        return items

    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "source": "Google News",
            "url": link
        })

    return items

def get_sentiment(ticker: str, max_items: int = 10):
    ticker = ticker.upper().strip()

    news_items = _yahoo_news(ticker, max_items=max_items)
    if not news_items:
        news_items = _google_news_rss(ticker, max_items=max_items)

    if not news_items:
        return 0.0, []

    scores = [analyzer.polarity_scores(item["title"])["compound"] for item in news_items]
    return float(sum(scores) / len(scores)), news_items