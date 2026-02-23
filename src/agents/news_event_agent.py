import os
import asyncio
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# set up the model connection using ChatGPT 4o mini as model
_model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# set up NewsAPI client
_news_event_client = NewsApiClient(
    api_key=os.getenv("NEWSAPI_KEY")
)

# set up agent, connect with model defined above
_news_event_agent = AssistantAgent(
    name="NewsEventAgent",
    model_client=_model_client,
    system_message=(
        "You are a financial news sentiment analyst.\n"
        "Given news articles JSON, analyze the overall sentiment regarding the market event.\n"
        "Return STRICT JSON only with keys:\n"
        "overall_sentiment (string: 'positive', 'negative', 'neutral'), "
        "confidence (0..1 number), "
        "key_articles (list of dicts with 'title', 'sentiment'), "
        "summary (string).\n"
        "Focus on how the news impacts the likelihood of the event outcome."
    ),
)

async def _fetch_news_articles(query: str, max_articles: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent news articles related to the query."""
    try:
        newsapi = _news_event_client
        # Get articles from the last 7 days
        from datetime import datetime, timedelta
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page_size=max_articles
        )
        
        articles = response.get('articles', [])
        # Extract relevant fields
        cleaned_articles = []
        for article in articles:
            cleaned_articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', '')
            })
        
        return cleaned_articles
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def _build_news_query(ctx: Dict[str, Any]) -> str:
    """Build a search query from market context."""
    title = ctx.get('title', '')
    event_ticker = ctx.get('event_ticker', '')
    
    # Use title as primary query, add event_ticker if available
    query = title
    if event_ticker and event_ticker != title:
        query += f" OR {event_ticker}"
    
    # Remove common market terms that might not help news search
    query = query.replace("Will ", "").replace("will ", "")
    query = query.replace("?", "").strip()
    
    return query


async def run_news_event_agent(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch news articles and analyze sentiment to determine trading signal.
    
    Returns a standardized AgentOutput dict.
    """
    # Build search query
    query = _build_news_query(ctx)
    
    # Fetch news articles
    articles = await _fetch_news_articles(query, max_articles=5)
    
    if not articles:
        return {
            "agent": "NewsEventAgent",
            "action": None,
            "direction": None,
            "score": 0.5,
            "reason": "No relevant news articles found.",
            "signals": {},
            "raw": {"query": query, "articles": []}
        }
    
    # Prepare articles for LLM analysis
    articles_json = json.dumps(articles, indent=2)
    market_title = ctx.get('title', 'Unknown Event')
    
    prompt = f"Market Event: {market_title}\n\nNews Articles:\n{articles_json}\n\nAnalyze the sentiment and return STRICT JSON only."
    
    # Run sentiment analysis
    try:
        sentiment_agent = _news_event_agent
        result = await sentiment_agent.run(task=[TextMessage(content=prompt, source="user")])
        text = result.messages[-1].content
        
        # Parse LLM response
        sentiment_data = json.loads(text)
        overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
        confidence = sentiment_data.get('confidence', 0.5)
        summary = sentiment_data.get('summary', '')
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        overall_sentiment = 'neutral'
        confidence = 0.5
        summary = "Analysis failed."
    
    # Convert sentiment to trading signal
    if overall_sentiment == 'positive':
        action = "BUY"
        direction = "YES"  # Assuming positive news favors YES
        score = confidence
        reason = f"Positive news sentiment: {summary}"
    elif overall_sentiment == 'negative':
        action = "BUY"
        direction = "NO"   # Negative news favors NO
        score = confidence
        reason = f"Negative news sentiment: {summary}"
    else:
        action = None
        direction = None
        score = 0.5
        reason = f"Neutral news sentiment: {summary}"
    
    return {
        "agent": "NewsEventAgent",
        "action": action,
        "direction": direction,
        "score": float(score),
        "reason": reason,
        "signals": {
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "articles_found": len(articles),
            "query": query
        },
        "raw": {
            "articles": articles,
            "sentiment_analysis": sentiment_data if 'sentiment_data' in locals() else {}
        }
    }

# need async function because the agent.run() makes a network call, need to wait
# for a response
def run_news_event_agent_sync(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(run_news_event_agent(ctx))
