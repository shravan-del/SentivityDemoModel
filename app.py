"""
Sentivity Demo API
Endpoint: POST /v1/demo
"""

import os
import re
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import praw
from flask import Flask, jsonify, request
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Credentials (from environment) ───────────────────────────────────────────
REDDIT_CLIENT_ID     = os.environ['REDDIT_CLIENT_ID']
REDDIT_CLIENT_SECRET = os.environ['REDDIT_CLIENT_SECRET']
REDDIT_USER_AGENT    = os.getenv('REDDIT_USER_AGENT', 'sentivity-demo/1.0')
OPENAI_API_KEY       = os.environ['OPENAI_API_KEY']

# ── Clients ───────────────────────────────────────────────────────────────────
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    check_for_async=False,
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
vader = SentimentIntensityAnalyzer()

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 30
POSTS_PER_BRAND = 50
NUM_COMPETITORS = 2

# ── Data Collection ───────────────────────────────────────────────────────────

def get_cutoff_ts() -> float:
    return (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp()


def fetch_posts(brand: str) -> list:
    cutoff = get_cutoff_ts()
    posts = []
    try:
        for sub in reddit.subreddit('all').search(
            query=brand, sort='relevance', time_filter='month', limit=POSTS_PER_BRAND
        ):
            if sub.created_utc < cutoff:
                continue
            posts.append({
                'text':        f"{sub.title} {sub.selftext}".strip(),
                'title':       sub.title,
                'score':       sub.score,
                'created_utc': sub.created_utc,
            })
    except Exception as e:
        logger.warning(f"Reddit error for '{brand}': {e}")
    return posts


def score_posts(posts: list) -> list:
    for p in posts:
        p['sentiment'] = vader.polarity_scores(p['text'])['compound']
    return posts


def mean_sentiment(posts: list):
    if not posts:
        return None
    return float(np.mean([p['sentiment'] for p in posts]))

# ── LLM Helpers ───────────────────────────────────────────────────────────────

def llm(system: str, user: str, temperature: float = 0.4) -> str:
    resp = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=temperature,
        max_tokens=300,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': user},
        ]
    )
    return resp.choices[0].message.content.strip()


def get_competitors(brand: str) -> list:
    raw = llm(
        system='You are a market research assistant. Return ONLY a comma-separated list of brand names — no explanation, no numbering.',
        user=f'List {NUM_COMPETITORS} direct consumer-facing competitors of "{brand}".'
    )
    return [c.strip().strip('"') for c in raw.split(',') if c.strip()][:NUM_COMPETITORS]


def extract_negative_quote(brand: str, posts: list) -> str:
    bottom = sorted(posts, key=lambda p: p['sentiment'])[:10]
    corpus = '\n---\n'.join(p['text'][:400] for p in bottom)
    return llm(
        system=(
            'You surface specific consumer complaints. '
            'Return ONE 1-2 sentence quote that sounds like a real consumer — '
            'name a specific feature, product, or experience. No generic category statements.'
        ),
        user=f'Brand: {brand}\n\nSource posts:\n{corpus}'
    )


def extract_wish_quote(brand: str, posts: list) -> str:
    WISH_RE = re.compile(
        r'\b(wish|want|would be nice|hope they|should add|need to add|missing|lacks?)\b',
        re.IGNORECASE
    )
    wish_posts = [p for p in posts if WISH_RE.search(p['text'])]
    source = wish_posts[:15] if wish_posts else posts[:15]
    corpus = '\n---\n'.join(p['text'][:400] for p in source)
    return llm(
        system=(
            'You surface product improvement requests. '
            'Return ONE 1-2 sentence first-person quote: "I wish [Brand] had/did [specific thing]." '
            'Be concrete — name the feature, flavor, product line, or capability.'
        ),
        user=f'Brand: {brand}\n\nSource posts:\n{corpus}'
    )


def generate_firm_insight(brand: str, pct_change: float, volatility: float, posts: list) -> str:
    sample = '\n---\n'.join(p['text'][:300] for p in posts[:20])
    return llm(
        system=(
            'You are a senior market intelligence analyst writing for a financial or advisory firm. '
            'Given sentiment metrics and raw consumer posts, write ONE specific, actionable insight '
            'the firm should share with its clients — 2-3 sentences. '
            'Reference a concrete trend, risk, or opportunity. No generic commentary.'
        ),
        user=(
            f'Brand: {brand}\n'
            f'Sentiment % change over {LOOKBACK_DAYS} days: {pct_change:+.1f}%\n'
            f'Sentiment volatility (std dev of daily scores): {volatility:.4f}\n\n'
            f'Sample consumer posts:\n{sample}'
        )
    )

# ── Mode Logic ────────────────────────────────────────────────────────────────

def run_company_mode(brand_name: str) -> dict:
    logger.info(f'[Company Mode] {brand_name}')

    brand_posts = score_posts(fetch_posts(brand_name))
    if not brand_posts:
        raise ValueError(f'No posts found for "{brand_name}" in the past {LOOKBACK_DAYS} days.')

    brand_avg = mean_sentiment(brand_posts)
    logger.info(f'{len(brand_posts)} posts | avg sentiment: {brand_avg:.4f}')

    competitors = get_competitors(brand_name)
    logger.info(f'Competitors: {competitors}')

    comp_avgs = {}
    for comp in competitors:
        comp_posts = score_posts(fetch_posts(comp))
        avg = mean_sentiment(comp_posts)
        if avg is not None:
            comp_avgs[comp] = avg
            logger.info(f'{comp}: {avg:.4f} ({len(comp_posts)} posts)')

    if not comp_avgs:
        raise ValueError('Could not retrieve data for any competitor.')

    comp_mean        = float(np.mean(list(comp_avgs.values())))
    competitor_delta = round((brand_avg - comp_mean) * 100, 2)

    negative_quote = extract_negative_quote(brand_name, brand_posts)
    wish_quote     = extract_wish_quote(brand_name, brand_posts)

    return {
        'mode':             'company',
        'brand':            brand_name,
        'competitor_delta': competitor_delta,
        'competitors':      list(comp_avgs.keys()),
        'negative_quote':   negative_quote,
        'wish_quote':       wish_quote,
    }


def run_firm_mode(brand_name: str) -> dict:
    logger.info(f'[Firm Mode] {brand_name}')

    posts = score_posts(fetch_posts(brand_name))
    if not posts:
        raise ValueError(f'No posts found for "{brand_name}" in the past {LOOKBACK_DAYS} days.')

    logger.info(f'{len(posts)} posts found')

    df = pd.DataFrame(posts)
    df['date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
    daily = df.groupby('date')['sentiment'].mean().sort_index()

    mid        = len(daily) // 2
    first_avg  = daily.iloc[:mid].mean() if mid > 0 else daily.mean()
    second_avg = daily.iloc[mid:].mean() if mid > 0 else daily.mean()

    if abs(first_avg) < 1e-6:
        sentiment_pct_change = 0.0
    else:
        sentiment_pct_change = round(((second_avg - first_avg) / abs(first_avg)) * 100, 2)

    volatility = round(float(daily.std()), 4) if len(daily) > 1 else 0.0

    logger.info(f'% change: {sentiment_pct_change:+.2f}% | volatility: {volatility:.4f}')

    client_insight = generate_firm_insight(brand_name, sentiment_pct_change, volatility, posts)

    return {
        'mode':                 'firm',
        'brand':                brand_name,
        'sentiment_pct_change': sentiment_pct_change,
        'volatility':           volatility,
        'client_insight':       client_insight,
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/v1/demo', methods=['POST'])
def demo():
    body = request.get_json(silent=True)

    if not body:
        return jsonify({'error': 'Request body must be JSON'}), 400

    brand_name = body.get('brand_name', '').strip()
    is_firm    = body.get('is_firm')

    if not brand_name:
        return jsonify({'error': 'brand_name is required'}), 400
    if not isinstance(is_firm, bool):
        return jsonify({'error': 'is_firm must be a boolean'}), 400

    try:
        if is_firm:
            result = run_firm_mode(brand_name)
        else:
            result = run_company_mode(brand_name)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        logger.exception('Unhandled error in /v1/demo')
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
