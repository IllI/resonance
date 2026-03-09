import urllib.request
import xml.etree.ElementTree as ET

try:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC11046465&retmode=xml"
    u = urllib.request.urlopen(url)
    xml_data = u.read()
    root = ET.fromstring(xml_data)
    
    print("Authors:")
    for contrib in root.findall('.//contrib[@contrib-type="author"]'):
        name = contrib.find('name')
        if name is not None:
            given = name.find('given-names')
            surname = name.find('surname')
            if given is not None and surname is not None:
                print(f" - {given.text} {surname.text}")
                
    print("\nEmails:")
    for email in root.findall('.//email'):
        print(f" - {email.text}")
except Exception as e:
    print(f"Error: {e}")
