import urllib.request
import xml.etree.ElementTree as ET

try:
    u = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=22869999&retmode=xml")
    root = ET.fromstring(u.read())
    for abstract in root.iter('AbstractText'):
        if abstract.text:
            print(abstract.text)
except Exception as e:
    print(e)
