import urllib.request
import xml.etree.ElementTree as ET

try:
    u = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC11046465&retmode=xml")
    xml_data = u.read()
    root = ET.fromstring(xml_data)
    for sec in root.iter('sec'):
        title = sec.find('title')
        if title is not None and ('Data' in title.text or 'Availability' in title.text or 'availability' in title.text):
            print(f"--- {title.text} ---".encode('ascii', 'ignore').decode('ascii'))
            for p in sec.iter('p'):
                if p.text:
                    print(p.text.strip().encode('ascii', 'ignore').decode('ascii'))
except Exception as e:
    print(f"Error: {e}")
