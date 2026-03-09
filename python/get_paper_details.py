import urllib.request
import xml.etree.ElementTree as ET

url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11039847&retmode=xml'
u = urllib.request.urlopen(url)
tree = ET.parse(u)
root = tree.getroot()

text = ''.join(root.itertext())
# Try to find the abstract and method sections
print("Abstract:")
abstract = text[text.find('Abstract'):text.find('Abstract')+2000].encode('ascii', errors='ignore').decode()
print(abstract)

print("\nMethod/EEG/fMRI mention:")
if "fMRI" in text or "functional magnetic resonance imaging" in text.lower():
    print("Found fMRI")
if "EEG" in text or "electroencephal" in text.lower():
    print("Found EEG")
