import requests
import json
# from main import ServerStart

def latestnews():
    APICollection = {"business":"https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=09172b8b482344c8991c442c9b283c45",
                     "entertainment":"https://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey=09172b8b482344c8991c442c9b283c45",
                     "health":"https://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=09172b8b482344c8991c442c9b283c45",
                     "science":"https://newsapi.org/v2/top-headlines?country=in&category=science&apiKey=09172b8b482344c8991c442c9b283c45",
                     "sports":"https://newsapi.org/v2/top-headlines?country=in&category=sports&apiKey=09172b8b482344c8991c442c9b283c45",
                     "technology":"https://newsapi.org/v2/top-headlines?country=in&category=technology&apiKey=09172b8b482344c8991c442c9b283c45",
                    #  "india":"https://newsapi.org/v2/top-headlines?country=in&apiKey=09172b8b482344c8991c442c9b283c45"
                     }
    
    content = None
    url = None

    field = input("Choose the category: \n1. business\n2. entertainment\n3. health\n4. science\n5. sports\n6. technology\n7. india\nEnter name of category:")

    # code to implement
    # say("I have searched for these category for country India: Business, Entertainment, Health, Science, Sports, Technology")
    # field = speech_to_text(input)

    for key, value in APICollection.items():
        if key.lower() in field.lower():
            url = value
            # say(news found for the category {field})
            print(url)
            print("URL Found!!!")
            break

        else:
            url = True
            if url is True:
                print("URL Not Found!!!")
                # say(news not found)

    news = requests.get(url).text
    news = json.loads(news)

    print("News:\n")
    arts = news["articles"]
    for articles in arts:
        article = articles["title"]
        print(article)
        # say(article)
        news_url = articles["url"]
        print(f"know more...\n{news_url}")

        a = input("To continue...Press 1\nTo stop...Press 2\nEnter choice:")
        # say("Do you want more news?")
        # a = speech_to_text(input - yes, no)
        if str(a) == "1":
            # str(a) == "yes"
            pass
        elif str(a) == "2":
            #str(a) == "no"
            break

# latestnews()


# try this approach
# response = news + know more and the continue condition
# conn.sendall(response.encode())