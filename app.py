import requests #for HTTP requests
import csv #to read and write to csv files
import pandas as pd #for analysing data
from textblob import TextBlob #for sentiment analysis 
import matplotlib.pyplot as plt #for creating histogram
import plotly.graph_objs as go #for creating pie charts
from flask import Flask, render_template, make_response, request #for building the web app  
import io #to handle file objects
from matplotlib.backends.backend_agg import FigureCanvas #to render histogram
import base64 #to convert the plot from matplotlib into an img tag for html
import plotly.express as px #for pie chart
import re #for data preprocessing

#creates a flask app named 'app'
app = Flask(__name__, template_folder='/home/kali/Desktop/project/templates', static_folder='/home/kali/Desktop/project/static') 

@app.route('/') #default end point
def index(): #defining the index function
    return render_template('index.html') #will display the message on index.html

@app.route('/sentiment', methods=['POST'])#post request for user input
def subreddit_sentiment():
    subreddit = request.form['subreddit']#subreddit name
    post_type = request.form['post_type']#top posts or new posts
    
    if post_type == 'top':
        url = f'https://www.reddit.com/r/{subreddit}/top.json?limit=100'#for retrieving top 100 posts
    elif post_type == 'new':
        url = f'https://www.reddit.com/r/{subreddit}/new.json?limit=100'#for retrieving 100 new posts
    else:
        # handle invalid post type input here
        return 'Invalid post type'
        
    headers = {'User-Agent': 'myBot/0.0.1'} #user agent header for requesting from API
    response = requests.get(url, headers=headers)
    data = response.json() #converts the response to JSON
    posts = [] #an empty list that gets 100 titles and permalinks
    with open('reddit_data.csv', 'w', newline='', encoding='utf-8') as f:#creating csv file
        writer = csv.writer(f)
        writer.writerow(['Title', 'Permalink'])#writes title and permalink
        for post in data['data']['children']:
            permalink = f"https://www.reddit.com/r/{subreddit}/comments/{post['data']['id']}/{post['data']['title'].replace(' ', '_')}/"
            posts.append([post['data']['title'], permalink])
            writer.writerow([post['data']['title'], permalink])
    reddit_data = pd.read_csv('reddit_data.csv')
    
    #sentiment analysis process 
    
    df = pd.read_csv('reddit_data.csv')# Read the CSV file and store it in a dataframe

    df['Title'] = df['Title'].apply(lambda x: re.sub('[^a-zA-Z\s]+', '', x))# Remove non-alphabetic characters

    df['Title'] = df['Title'].apply(lambda x: x.lower())# Convert to lowercase

    df['Title'] = df['Title'].apply(lambda x: ' '.join([word.lemmatize() for word in TextBlob(x).words]))# Perform lemmatization using TextBlob

    df.to_csv('cleaned_data.csv', index=False)# Save the cleaned data to a new CSV file
    

    sentiments = [] #creates a list
    for post in reddit_data['Title']:#textblob analyses each post
        text = TextBlob(post)
        sentiment = text.sentiment.polarity#checks sentiment polarity of text
        sentiments.append(sentiment)#appends to the sentiment list
    df['Sentiment'] = sentiments #sentiment column is added to the dataframe

    # calculate the percentage breakdown
    positive = df['Sentiment'][df['Sentiment']>0].count() / df['Sentiment'].count()
    negative = df['Sentiment'][df['Sentiment']<0].count() / df['Sentiment'].count()
    neutral = 1 - positive - negative

    sentiment_df = df[['Title', 'Sentiment', 'Permalink']]#new dataframe created with title, sentiment and permalink
    sentiment_df = sentiment_df.rename(columns={'Title': 'Post', 'Sentiment': 'Sentiment Score'})
    #renaming title to post and sentiment to sentiment scores
    sentiment_df = sentiment_df[['Post', 'Sentiment Score', 'Permalink']]#reorders the columns
    
    sentiment_table = sentiment_df.to_html(classes="table table-bordered table-striped", index=False)# Convert dataframe to HTML table
    
    labels = ['Positive', 'Negative', 'Neutral'] #assigns labels
    values = [positive, negative, neutral] #assigns values
    colors = ['#2ca02c', '#d62728', '#7f7f7f'] #assigns colors to each value
    fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])#creates the chart
    fig2.update_traces(marker=dict(colors=colors))
    fig2.update_layout(title=f'Percentage Breakdown of Sentiments for r/{subreddit}',#title of pie chart
    		  title_x=0.5,
                  title_font_size=18,
                  title_font_color='black',
                  height=600)
    
    fig, ax = plt.subplots(figsize=(7, 6)) #Create histogram using matplotlib
    df['Sentiment'].plot(kind='hist', bins=10, range=(-1, 1), ax=ax)
    ax.set_xlabel('Sentiment Score')#x axis
    ax.set_ylabel('Number of Posts')#y axis
    ax.set_title(f'Sentiment Analysis of Reddit Posts from r/{subreddit}')#title of histogram
    
    canvas = FigureCanvas(fig)#creates a FigureCanvasobject
    output = io.BytesIO()#stores image in memory
    canvas.print_png(output)#converts the histogram to a png format
    histogram_url = 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()#assigned to histogram_url

    fig2_bytes = io.BytesIO()#stores image in memory
    fig2.write_image(fig2_bytes, format='png')#converts pie chart to png format
    fig2_bytes.seek(0)
    fig2_base64 = base64.b64encode(fig2_bytes.read()).decode('ascii')#to embed in the html page
    pie_chart_url = 'data:image/png;base64,{}'.format(fig2_base64)#assigned to pie_chart_url
    
    fig1_bytes = io.BytesIO()# Generate image URLs for both plots
    FigureCanvas(fig).print_png(fig1_bytes)
    fig1_bytes.seek(0)
    fig1_base64 = base64.b64encode(fig1_bytes.read()).decode('ascii')
    histogram_url = 'data:image/png;base64,{}'.format(fig1_base64)
    
    #returns index.html with histogram, table, pie chart, sentiment analysis and table
    return render_template('index.html', message='Reddit Sentiment Analysis', show_plot=True, histogram=histogram_url, pie_chart=pie_chart_url, sentiment_table=sentiment_table, sentiment_data=sentiment_df.to_dict('records'))

