import urllib.request
import urllib.parse
import json

base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=20'

queries = [
    'fMRI AND ("distant intentionality" OR "distant mental interactions" OR "anomalous information reception" OR "anomalous cognition" OR "non-local consciousness" OR "quantum consciousness" OR "telepathy" OR "psi" OR "extended mind")',
    'fMRI AND ("mindsight" OR "clairvoyance" OR "remote viewing" OR "extra-sensory" OR "extrasensory")',
    '"functional magnetic resonance imaging" AND ("distant intentionality" OR "anomalous cognition" OR "non-local consciousness")'
]

for q in queries:
    url = f"{base_url}&term={urllib.parse.quote(q)}"
    print(f"Query: {q}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            ids = data.get('esearchresult', {}).get('idlist', [])
            print(f"IDs found: {ids}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 20)
