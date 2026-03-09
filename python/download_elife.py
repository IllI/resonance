import requests
import json

url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.7554/eLife.59360&format=json&resultType=core'
r = requests.get(url).json()

pmcid = r['resultList']['result'][0]['pmcid']
print("FOUND PMCID:", pmcid)

ft = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
r2 = requests.get(ft)
with open('elife.xml', 'w', encoding='utf-8') as f:
    f.write(r2.text)
