import urllib.request
import xml.etree.ElementTree as ET
try:
    u = urllib.request.urlopen('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=38660011&retmode=xml')
    root = ET.fromstring(u.read())
    title = root.find('.//ArticleTitle')
    print("TITLE: ", title.text if title is not None else "No Title")
    for a in root.iter('AbstractText'):
        print("ABSTRACT: ", a.text)
except Exception as e:
    print(e)
