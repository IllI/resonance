import urllib.request
import xml.etree.ElementTree as ET

try:
    u = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=3144613&retmode=xml")
    root = ET.fromstring(u.read())
    for p in root.findall('.//sec/p'):
        text = "".join(p.itertext()).lower()
        if 'success' in text or 'correct' in text or 'reproduced' in text or 'image' in text:
            print(f"--- PARAGRAPH ---")
            print("".join(p.itertext()).strip())
except Exception as e:
    print(f"Error: {e}")
