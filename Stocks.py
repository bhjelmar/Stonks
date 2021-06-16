import json
import time
from datetime import datetime, timedelta
from itertools import islice
from typing import List

import psaw
from numpy import interp
from psaw import PushshiftAPI
import requests
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import matplotlib.pyplot as plt

import logging


def getTickerInfo() -> List[str]:
    r = requests.get("https://api.swaggystocks.com/wsb/sentiment/top?limit=500")

    def take(n, iterable):
        return list(islice(iterable, n))

    return take(2, {ticker['ticker']: ticker for ticker in json.loads(r.text)})


def getCommentsFromSubreddit(subreddit: str, query: str, before: datetime, after: datetime):
    api = PushshiftAPI()

    gen = api.search_comments(
        subreddit=subreddit,
        q=query,
        filter=['body', 'score', 'subreddit'],
        sort_type='score',
        limit=100,
        after=int(after.timestamp()),
        before=int(before.timestamp()),
    )
    return list(gen)


def findCommentsWhichReferenceTicker(comments, ticker):
    commentsWhichReferenceTicker = []
    for comment in comments:
        words = comment.body.split()
        upper_words = []
        for word in words:
            if all(c.isupper() for c in word):
                upper_words.append(word)

        for symbol in upper_words:
            if symbol == ticker:
                commentsWhichReferenceTicker.append(comment)
    return commentsWhichReferenceTicker


def analyzeComments(comments) -> float:
    sia = SentimentIntensityAnalyzer()
    if len(comments) > 0:
        polarity_scores = []
        for comment in comments:
            polarityScore = sia.polarity_scores(comment.body)["compound"]
            polarityScore *= comment.score
            polarity_scores.append(polarityScore)
        return mean(polarity_scores) * len(comments)
    else:
        return 0


def main():
    tickers = getTickerInfo()

    lookbackDays = 30

    now = datetime.now()
    today = datetime(now.year, now.month, now.day)
    start = datetime(today.year, today.month, today.day) - timedelta(lookbackDays)
    end = start + timedelta(1)
    data = yf.download(tickers, start=start, end=now)

    tickerPlot = {ticker: ([], []) for ticker in tickers}
    while start != today:
        print(start)
        print(end)
        print('================')
        start += timedelta(1)
        end += timedelta(1)

        for ticker in tickers:
            comments = getCommentsFromSubreddit(
                subreddit='investing',
                query=ticker,
                after=start,
                before=end)
            commentsWhichReferenceTicker = findCommentsWhichReferenceTicker(comments, ticker)
            meanCommentScore = analyzeComments(commentsWhichReferenceTicker)
            tickerPlot[ticker][0].append(start)
            tickerPlot[ticker][1].append(meanCommentScore)

    for ticker in tickers:
        fig, ax = plt.subplots()

        dates = [datetime.utcfromtimestamp(timestamp / 1_000_000_000) for timestamp in data.index.values.tolist()]
        ax.plot(dates, data["Adj Close"][ticker].tolist(), color='red')
        ax.set_ylabel("Stock Price", color="red", fontsize=14)

        ax2 = ax.twinx()
        xVals = tickerPlot[ticker][0]
        yVals = tickerPlot[ticker][1]
        ax2.plot(xVals, yVals, color='blue')
        ax2.set_ylabel("Stock Sentiment", color="blue", fontsize=14)

        for i in range(0, len(xVals) - 1):
            if yVals[i] > 0:
                color = (0, int(interp(yVals[i], [1, max(yVals)], [0, 1])), 0, 1.0)
            elif yVals[i] < 0:
                color = (int(interp(yVals[i], [1, min(yVals)], [0, 1])), 0, 0, 1.0)
            else:
                color = (0, 0, 0, 1.0)
            ax.axvspan(xVals[i], xVals[i + 1], alpha=0.5, color=color, edgecolor=None)

        fig.savefig(f'out/{ticker}.png')


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    logger = logging.getLogger('psaw')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    main()
