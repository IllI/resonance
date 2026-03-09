import requests

pmcid = 'PMC7755353'
url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
r = requests.get(url)

with open('paper_full_text.xml', 'w', encoding='utf-8') as f:
    f.write(r.text)

print("Downloaded PMC7755353 XML.")
