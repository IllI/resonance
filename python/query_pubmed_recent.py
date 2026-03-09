import urllib.request
import urllib.parse
import json

ids = ['41755935', '41688809', '41681784', '41678144', '41645742', '39924830', '36581541']
url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={','.join(ids)}"

try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        result = data.get('result', {})
        for uid in result.get('uids', []):
            item = result.get(uid, {})
            title = item.get('title', '')
            pubdate = item.get('pubdate', '')
            authors = [a.get('name', '') for a in item.get('authors', [])]
            print(f"[{uid}] {pubdate} | {title} | Authors: {', '.join(authors)}")
except Exception as e:
    print(f"Error: {e}")
