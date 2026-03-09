import urllib.request
import xml.etree.ElementTree as ET

try:
    u = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=24427128,22869999,21829287,18095790,16398587")
    root = ET.fromstring(u.read())
    for docsum in root.iter('DocSum'):
        title_item = docsum.find('.//Item[@Name="Title"]')
        if title_item is not None:
            print(title_item.text)
except Exception as e:
    print(e)
