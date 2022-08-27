
import secrets_reddit as sr #Updated to correct file, with an alias
import random

from typing import Dict, List

from praw import Reddit
from praw.models.reddit.subreddit import Subreddit
from praw.models import MoreComments

from transformers import pipeline


def get_subreddit(display_name:str) -> Subreddit:
    """Get subreddit object from display name

    Args:
        display_name (str): [description]

    Returns:
        Subreddit: [description]
    """
    reddit = Reddit( #Updated to new aliased file
        client_id = sr.REDDIT_API_CLIENT_ID,
        client_secret = sr.REDDIT_API_CLIENT_SECRET,
        user_agent = sr.REDDIT_API_USER_AGENT
    )
    
    subreddit = reddit.subreddit(display_name)#Updated to work with method variable
    return subreddit

def get_comments(subreddit:Subreddit, limit:int=3) -> List[str]:
    """ Get comments from subreddit

    Args:
        subreddit (Subreddit): [description]
        limit (int, optional): [description]. Defaults to 3.

    Returns:
        List[str]: List of comments
    """
    top_comments = []
    for submission in subreddit.top(limit=limit):
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            top_comments.append(top_level_comment.body)
    return top_comments

def run_sentiment_analysis(comment:str) -> Dict:
    """Run sentiment analysis on comment using default distilbert model
    
    Args:
        comment (str): [description]
        
    Returns:
        str: Sentiment analysis result
    """
    sentiment_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")#Updated to bertweet model
    sentiment = sentiment_model(comment)
    return sentiment[0]


if __name__ == '__main__':
    subreddit = get_subreddit('TSLA')#Updated to instantiate with TSLA
    comments = get_comments(subreddit)
    comment = random.choice(comments)#Updated to randomly select a comment from the comments list
    sentiment = run_sentiment_analysis(comment)
    
    print(f'The comment: {comment}')
    print(f'Predicted Label is {sentiment["label"]} and the score is {sentiment["score"]:.3f}')
