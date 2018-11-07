# Crawl and Visualize ICLR 2019 OpenReview Data

<p align="center">
    <img src="asset/logo_wordcloud.png" width="720"/>
</p>

## Descriptions

This Jupyter Notebook contains the data and visualizations that are crawled ICLR 2019 OpenReview webpages. As some are the reviews are still missing (11.3299\% by the time the data is crawled), the results might not be accurate. 

## Visualizations 


The word clouds formed by keywords of submissions show the hot topics including **reinforcement learning**, **generative adversarial networks**, **generative models**, **imitation learning**, **representation learning**, etc.
<p align="center">
    <img src="asset/wordcloud.png" width="720"/>
</p>

This figure is plotted with python [word cloud generator](https://github.com/amueller/word_cloud) 

```
from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=64, max_words=160, 
                      width=1280, height=640,
                      background_color="black").generate(' '.join(keywords))
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

The distributions of reviewer ratings center around 5 to 6 (mean: 5.15).

<p align="center">
    <img src="asset/rating.png" width="640"/>
</p>

You can compute how many papers are beaten by yours with

```
def PR(rating_mean, your_rating):
    pr = np.sum(your_rating >= np.array(rating_mean))/len(rating_mean)*100
    return pr
my_rating = (7+7+9)/3  # your average rating here
print('Your papar beats {:.2f}% of submission '
      '(well, jsut based on the ratings...)'.format(PR(rating_mean, my_rating)))
# ICLR 2017: accept rate 39.1% (198/507) (15 orals and 183 posters)
# ICLR 2018: accept rate 32% (314/981) (23 orals and 291 posters)
# ICLR 2018: accept rate ?% (?/1580)
```

The top 50 common keywrods and their frequency.

<p align="center">
    <img src="asset/frequency.png" width="640"/>
</p>

The average reviewer ratings and the frequency of keywords indicate that to maximize your chance to get higher ratings would be using the keyowrds such as **theory**, **robustness**, or **graph neural network**.

<p align="center">
    <img src="asset/rating_frequency.png" width="800"/>
</p>

## How it works

To crawl data from dynamic websites such as OpenReview, a headless web simulator is created by

```
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
executable_path = '/Users/waltersun/Desktop/chromedriver'  # path to your executable browser
options = Options()
options.add_argument("--headless")
browser = webdriver.Chrome(options=options, executable_path=executable_path)  
```

Then, we can get the content of a webpage

```
browser.get(url)
```

To know what content we can crawl, we will need to inspect the webpage layout.

<p align="center">
    <img src="asset/inspect.png" width="720"/>
</p>

I chose to get the content by

```
key = browser.find_elements_by_class_name("note_content_field")
value = browser.find_elements_by_class_name("note_content_value")
```

The data includes the abstract, keywords, TL; DR, comments.
