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
        
        if confidence*100 >= 80:
            output1 = open("python1.json","a")
            output1.write(tweets)
            output1.close()


            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.close()            
            
            
        return True
    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, Mylistener())
twitterStream.filter(track=["protest", "attack", "police"], languages=["en"])
