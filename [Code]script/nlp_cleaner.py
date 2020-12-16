# import the modules
import re
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from collections import OrderedDict 
from wordcloud import WordCloud, STOPWORDS
from flask import render_template_string, jsonify

def data_clean(path):
    df = pd.read_csv(path,encoding = "ISO-8859-1")
    df['OriginalTweet']=df['OriginalTweet'].astype(str)
    df['Sentiment']=df['Sentiment'].astype(str)
    df['TweetAt']=df['TweetAt'].astype(str)
    df['TweetAt'] = pd.to_datetime(df['TweetAt'] , format='%d-%m-%Y')
    df['hash']=df['OriginalTweet'].apply(lambda x:extract_hashtags(x))
    df['OriginalTweet']=df['OriginalTweet'].apply(lambda x:clean_text(x))
    return df  
# function to print all the hashtags in a text 
def extract_hashtags(text): 
    text = text.lower()
    # the regular expression 
    regex = "#(\w+)" 
    # extracting the hashtags 
    hashtag_list = re.findall(regex, text)
    return ' '.join(hashtag_list)


def clean_text(text,remove_stopwords=True):
    '''Text Preprocessing '''
    
    # Convert words to lower case
    text = text.lower()    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # remove stopwords
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Tokenize each word
    text =  nltk.WordPunctTokenizer().tokenize(text)
    
    # Lemmatize each token
    lemm = nltk.stem.WordNetLemmatizer()
    text = [lemm.lemmatize(word) for word in text]
    
    return ' '.join(text)

def plotly_wordcloud(df,date = -1):
    """A wonderful function that returns figure data for three equally
    wonderful plots: wordcloud, frequency histogram and treemap"""
    if date == -1:
        text = " ".join(df["OriginalTweet"].tolist())
    # join all documents in corpus
    else:
        text = " ".join(df[df.TweetAt==date]["OriginalTweet"].tolist())
    #text = " ".join(df["OriginalTweet"].tolist())
    
    word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(int(i[0]))
        y_arr.append(int(i[1]))

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(int(i * 80))

    trace = {
        "x":x_arr,
        "y":y_arr,
        "xaxis": 'x2',
        "yaxis": 'y2',
        "marker":{"size":new_freq_list,"color":color_list},
        "textfont":dict(size=new_freq_list, color=color_list),
        "hoverinfo":"text",
        "textposition":"top center",
        "hovertext":["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        "mode":"text",
        "text":word_list}
    
    if date == -1:
        text_hash = " ".join(df["hash"].tolist())
    # join all documents in corpus
    else:
        text_hash = " ".join(df[df.TweetAt==date]["hash"].tolist())
    #stop_word = stopwords.words('english')
    text_hash = text_hash.split()
    word = Counter(text_hash)
    word = OrderedDict(sorted(word.items(),key=lambda kv: kv[1],reverse = True)) 
    #print(list(word.values())[:20])
    #print(list(word.keys())[:20])
    trace2 = {
        "x":list(word.values())[:20],
        "y":list(word.keys())[:20],
        "type":"bar",
        "orientation":"h",
        "name":"",
        "marker": {"color": 'rgb(105, 116, 184)'
  }}

    return jsonify([trace2,trace])
