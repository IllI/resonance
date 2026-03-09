import json
import requests

url = "https://openneuro.org/graphql"
query = """
{
  datasets(first: 1000) {
    edges {
      node {
        id
        name
      }
    }
  }
}
"""

response = requests.post(url, json={'query': query})
if response.status_code == 200:
    try:
        data = response.json()
        if 'data' in data and 'datasets' in data['data']:
            datasets = data['data']['datasets']['edges']
            for d in datasets:
                name = d['node']['name'].lower() if d['node']['name'] else ""
                if 'medit' in name or 'mindful' in name or 'zen' in name:
                    print(d['node']['id'], d['node']['name'])
        else:
            print("Response did not contain expected fields:", data)
    except Exception as e:
        print("JSON Decode Error:", e)
else:
    print(f"Failed with {response.status_code}: {response.text}")
