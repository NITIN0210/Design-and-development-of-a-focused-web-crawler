from flask import Flask, render_template, request, url_for, flash, redirect
from selenium.webdriver import Chrome, Firefox
from selenium.common.exceptions import NoSuchElementException

from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import pandas as pd
import sqlite3
from flask import Flask, request, jsonify

# Function to create the database and table
def create_Seed_database():
    conn = sqlite3.connect('mydatabase_Seeds.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS seedTable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Content TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to update the database with new data
def update_Seed_database(name, content):
    conn = sqlite3.connect('mydatabase_Seeds.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO seedTable (name, content) VALUES (? ,?)', (name, content))
    conn.commit()
    conn.close()
def create_Docs_database():
    conn = sqlite3.connect('mydatabase_Docs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS docsTable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_name TEXT,
            link TEXT,
            content TEXT,
        )
    ''')
    conn.commit()
    conn.close()

# Function to update the database with new data
def update_Docs_database(seed_name, link, content):
    conn = sqlite3.connect('mydatabase_Docs.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO docsTable (seed_name, link, Content) VALUES (?, ?, ?)', (seed_name, link, content))
    conn.commit()
    conn.close()

# Initialize an empty DataFrame or load an existing DataFrame
# For example, if you want to initialize an empty DataFrame:
global df_seeds
df_seeds = pd.DataFrame(columns=['Name', 'Content'])

# Function to append data to the DataFrame
def append_seed_data(name, content):
    new_data = {
        'Name': name,
        'Content': content
    }
    new_df = pd.DataFrame([new_data])  # Create a DataFrame with the new data
    df = pd.concat([df_seeds, new_df], ignore_index=True)  # Append the new DataFrame to the original DataFrame
    return df

global df_docs
df_docs = pd.DataFrame(columns=['seed_name','link', 'content'])
def append_docs_data(seed_name,link, content):
    new_data = {
        'seed_name' : seed_name,
        'link' : link,
        'content' : content
    }
    new_df = pd.DataFrame([new_data])  # Create a DataFrame with the new data
    df = pd.concat([df_docs, new_df], ignore_index=True)  # Append the new DataFrame to the original DataFrame
    return df

data = {0:{'link':'links','paragrahs':'paragraphs','similarity_score':10}}
data_all_searches = {'test':[{0:{'link':'links','paragrahs':'paragraphs','similarity_score':10}}]}
search_texts = []
seed_pages_data = []
# Function to calculate similarity scores

documents_list = []
app = Flask(__name__)
messages = []

@app.route('/',methods=('GET','POST'))
def index():
    #take user input to search documents
    if request.method == 'POST':
        title = request.form['title']
        if not title:
            flash('Title is required!')
        else:
            messages.append({'title': title})
        driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
        #driver.get('https://en.wikipedia.org/wiki/Main_Page')
        driver.get('https://www.wikipedia.org/')
        driver.find_element_by_name('search').send_keys(title)
        search_button=driver.find_element_by_xpath('/html/body/div[3]/form/fieldset/button/i')
        search_button.click()
        print(driver)
        search_texts.append(title)

        query_para_list=[]
        for i in range(1,50):
            try:
                Main_doc_paras = driver.find_element_by_xpath('/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/p['+str(i)+']')
            except:
                continue
            print(Main_doc_paras.text)
            query_para_list.append(Main_doc_paras.text)
            x=''
            query_paragraphs=x.join(query_para_list)
        seed_pages_data.append(query_paragraphs)
        ## write code to store this data in df called query data where columns are query name, query content

        df_seedData = append_seed_data(title, seed_pages_data)
        print(df_seedData)
        create_Seed_database()
        update_Seed_database(title, seed_pages_data)

        ## end here

        # obtaining links from the seed page  into links list
        links=[]
        c=0
        for i in range(1,10):
            for j in range(1,5):
                if c==20:
                    break
                try:
                    link = driver.find_element_by_xpath('/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/p['+str(i)+']/a['+str(j)+']').get_attribute('href')
                except:
                    #continue
                    link = 'none'
                    print('error')
                print(i,j,link)
                links.append(link)
            c=c+1
        print(links,len(links))

        ## ende link extraction here
        #store the documents on a data frame and download to a database
        all_docs = []
        create_Docs_database()

        for a, i in enumerate(links):
            if i == 'none':
                continue
            driver.get(i)

            paragraphs_list = []
            data_link = {}
            for j in range(1,50):
                try:
                    paras = driver.find_element_by_xpath('/html/body/div[2]/div/div[3]/main/div[3]/div[3]/div[1]/p['+str(j)+']')
                except Exception as error:
                    #print('e',error)
                    continue
                paragraphs_list.append(paras.text)
            x=''
            paragraph=x.join(paragraphs_list)
            
            data_link['paragrahs']=paragraph
            documents_list.append(paragraph)
            data_link['link'] = i
            data_link['similarity_score'] = 10
            data[a]=data_link
            ## upload the data into a dataframe with column name as seed document name

            append_docs_data(title, i, paragraph)
            update_Seed_database(title, i, paragraph)
        #all_docs.append(data)
        data_all_searches[title]=([data])
        
        ## upload end

    
        ## take the documents and print out the bert models embedings similarity score and 
        # Load the pre-trained BERT model and tokenizer

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        #Main_doc_paras=''
        # Example query and related websites data dictionary

        try:
            # Attempt to access the variable
            print(query_paragraphs)
        except NameError:
        # Variable is not defined, print an error message
            print("Error: query_paragraphs is not initialized before use.")
            return redirect(url_for('index',mesg = messages, data={}))


        query = query_paragraphs
        related_websites_data = {}
        for id,doc in data.items():
            related_websites_data[id]=doc['paragrahs']


        # Tokenize the query and preprocess the query tensor
        query_tokens = tokenizer.encode(query, add_special_tokens=True, max_length=512, truncation=True)
        query_tensor = torch.tensor([query_tokens])

        # Preprocess each document (website data) in the dictionary
        document_tensors = {}
        for website_id, website_data in related_websites_data.items():
            website_tokens = tokenizer.encode(website_data, add_special_tokens=True, max_length=512, truncation=True)
            document_tensors[website_id] = torch.tensor([website_tokens])

        # Get embeddings for the query and documents
        with torch.no_grad():
            query_embedding = model(query_tensor)[0].mean(dim=1)  # Use mean pooling to get the query embedding
            document_embeddings = {website_id: model(doc_tensor)[0].mean(dim=1) for website_id, doc_tensor in document_tensors.items()}

        # Calculate cosine similarity between the query and each document
        similarities = {website_id: cosine_similarity(query_embedding, doc_embedding).item() for website_id, doc_embedding in document_embeddings.items()}

        # Sort documents based on similarity scores
        sorted_documents = [website_id for website_id, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)]

        # Print ranked list of websites based on relevance to the query
        print("Query:", query)
        print("Ranked List of Websites:")
        c=0
        for website_id in sorted_documents:
            if c==7:
                break
            #print(f"Website ID: {website_id}, Relevance Score: {similarities[website_id]:.4f}, Website Data: {related_websites_data[website_id]}")
            print(f"Website ID: {website_id}, Relevance Score: {similarities[website_id]:.4f},Link:{data[website_id]['link']}")
            c=c+1
        website_id 
        for id in sorted_documents:
            data[id]['similarity_score'] = similarities[id]
        
        driver.close()
        data_all_searches = {}
    # then keep a threshold and print related documents ans then also proceed to print summary using any summary model

    ## end summary model
        return redirect(url_for('index',mesg = messages))


    return render_template('index.html',mesg = messages, data=data)
@app.route('/<keyword>')
def searchLinks(keyword):
    return 'Searching for %s' % keyword
if __name__ == '__main__':
    app.run(debug=True)