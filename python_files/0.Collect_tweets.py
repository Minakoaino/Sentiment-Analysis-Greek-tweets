#!/usr/bin/env python
# coding: utf-8

import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import datetime
import csv
import pandas as pd



consumer_key = 'zRLze0lH2AKxY1bkVqpXfuhDZ'
consumer_secret = 'hAbqGAiTjwvTVBdUX0MMdl7Eqcb5eQPPY5jwx4SkGj07I4hGaD'
access_token = '1397618616653209602-3LUD2QVsxS5xjznEcTQJ6545G2lUko'
access_token_secret = 'DALjzoSMrU71FoULD4VOYN46yCtl9c1xsKghGtQGPvNxa'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)



class MyStreamListener(tweepy.StreamListener):
    def on_status(self,status):
        if hasattr(status, "retweeted_status"):
            try:
                current_status = str(status.retweeted_status.extended_tweet["full_text"])
                current_status = current_status.replace('\n', ' ').replace('\r', '')
                print(current_status)
                
            except AttributeError:
                current_status = str(status.retweeted_status.text)
                current_status = current_status.replace('\n', ' ').replace('\r', '')
                print(current_status)
        
        else:
            try:
                current_status = str(status.extended_tweet["full_text"])
                current_status = current_status.replace('\n', ' ').replace('\r', '')
                print(current_status)
            
            except AttributeError:
                current_status = str(status.text)
                current_status = current_status.replace('\n', ' ').replace('\r', '')
                print('Gathering tweets for hashtag - ', current_status)


                           
            csvw.writerow([status.id,
                           status.user.screen_name,
                           # created_at is a datetime object, converting to just grab the month/day/year
                           status.created_at.strftime('%m/%d/%y'),
                           status.favorite_count,
                           status.user.followers_count,
                           status.source,
                           status.user.location,
                           current_status])
            
    
    def on_error(self,status_code):
        if status_code == 420:
            print('You have been rate-limited for making too many requests')
            return False




if __name__ == '__main__':

    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = MyStreamListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = tweepy.Stream(auth, l)

    # Filter based on listed items
    csvw = csv.writer(open("5augoust", "a"))
    csvw.writerow(['twitter_id', 'name', 'created_at',
                   'followers_count', 'source', 'region', 'text'])
    stream.filter(track=['εμβολιο','εμβόλιο', 'εμβολιασμος', 'αντιεμβολιαστες', 'αντιεμβολιαστές', '#ανεμβολιαστοι'])


