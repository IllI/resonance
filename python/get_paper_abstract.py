import urllib.request
import xml.etree.ElementTree as ET

url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11039847&retmode=xml'
try:
    u = urllib.request.urlopen(url)
    tree = ET.parse(u)
    root = tree.getroot()
    abstract = root.find('.//abstract')
    if abstract is not None:
        print(''.join(abstract.itertext()).encode('ascii', 'ignore').decode())
    else:
        print("No abstract tag found")
except Exception as e:
    print(e)
