import urllib.request
import xml.etree.ElementTree as ET

try:
    u = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC11046465&retmode=xml")
    root = ET.fromstring(u.read())
    for name in root.findall('.//name'):
        surname = name.find('surname')
        given = name.find('given-names')
        if surname is not None and given is not None:
            print(f"{given.text} {surname.text}")
except Exception as e:
    print(f"Error: {e}")
