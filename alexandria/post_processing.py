import json
import requests
import urllib
import warnings

def get_api_key(filename='api_key.txt'):
    with open(filename) as f:
        return f.readline()

def search_book(input_text, api_key, n_results=5):
    input_text_enc = urllib.parse.quote(input_text)
    
    query = f'https://www.googleapis.com/books/v1/volumes?q={input_text_enc}&key={api_key}'
    response = requests.get(query)

    if response.status_code != 200:
        return None

    json = response.json()
    n_items = json.get('totalItems')
    if n_items == 0:
        warnings.warn(f'No items returned from the API.')
        return None
    
    titles = []
    for i in range(min(n_items, n_results)):
        item = json.get('items')[i]
        volume_info = item.get('volumeInfo')
        title = volume_info.get('title')

        titles.append(title)

    return titles 
    

if __name__ == '__main__':
    # user input
    input_text = 'fastai deep learning'
    api_key = get_api_key('api_key.txt')

    titles = search_book(input_text, api_key)

    print(titles)
    
