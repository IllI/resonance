import urllib.request
import xml.etree.ElementTree as ET

url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=6992642&retmode=xml'
try:
    response = urllib.request.urlopen(url)
    xml_data = response.read()
    root = ET.fromstring(xml_data)
    
    # Extract methods
    for sec in root.iter('sec'):
        title = sec.find('title')
        if title is not None and ('Method' in title.text or 'Material' in title.text or 'Experiment' in title.text):
            print(f"--- {title.text} ---")
            for p in sec.iter('p'):
                if p.text:
                    print(p.text.strip())
            print("\n")
except Exception as e:
    print("Error:", e)
