from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import json

def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        print('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)

results = open("results.html", "r").read()
links=open("links.txt", "w")
soup = BeautifulSoup(results, 'html.parser')
trs = soup.select("#query_result_main tr")
for tr in trs:
    tds = tr.select("td")
    if not tds: continue
    sell_date = tds[2].get_text()
    link = tds[3].get_text()
    y, m, d = map(int, sell_date.split('-'))
    if 2010 <= y <= 2018:
        print(link)
        cnt = simple_get('http://'+link+"&gc=gc")
        soup_cnt = BeautifulSoup(cnt, 'html.parser')
        imgs = soup_cnt.select('img[style="border:none;"]')
        print(imgs)
        for img in imgs:
            links.write("http://www.getchu.com/"+img.attrs['src']+'\n')
            links.flush()
