from tweepy import Stream
from tweepy.streaming import StreamListener
import json
from tweepy import OAuthHandler
import sentimentmodul as sen_mod



from Twitteraccount import *

class Mylistener (StreamListener):
    def on_data (self, data):
#        print(data)
#        return(True)
#    def on_error(self, data):
#        print(status)
        all_data = json.loads(data)

        tweets = all_data["text"] 
        sentiment_value, confidence = sen_mod.sentiment(tweets)
        print(tweets, sentiment_value, confidence)


#auth = OAuthHandler(ckey, csecret)
#auth.set_access_token(atoken, asecret)
#
#twitterStream = Stream(auth, Mylistener())
#twitterStream.filter(track=["protest"])
