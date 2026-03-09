import urllib.request
import xml.etree.ElementTree as ET

try:
    u = urllib.request.urlopen('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=3144613&retmode=xml')
    xml_data = u.read()
    root = ET.fromstring(xml_data)
    for sec in root.iter('sec'):
        title = sec.find('title')
        if title is not None and ('Method' in title.text or 'Material' in title.text or 'Experiment' in title.text):
            print(f"--- {title.text} ---")
            for p in sec.iter('p'):
                if p.text:
                    print(p.text.strip())
            print("\n")
except Exception as e:
    print(f"Error: {e}")
