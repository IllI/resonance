import urllib.request
from bs4 import BeautifulSoup
import re

url = "https://www.frontiersin.org/articles/10.3389/fpsyg.2019.03064/full"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')
    for h2 in soup.find_all('h2'):
        if 'Materials and Methods' in h2.text or 'Methods' in h2.text:
            curr = h2.find_next_sibling()
            while curr and curr.name != 'h2':
                print(curr.text)
                curr = curr.find_next_sibling()
            break
except Exception as e:
    print(e)
