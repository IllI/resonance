import xml.etree.ElementTree as ET

tree = ET.parse('paper_full_text.xml')
root = tree.getroot()

def get_text(element):
    return "".join(element.itertext())

# Find the Methods section
for sec in root.iter('sec'):
    title = sec.find('title')
    if title is not None and ('Method' in title.text or 'Procedure' in title.text or 'Task' in title.text):
        print(f"\n--- {title.text} ---")
        for p in sec.iter('p'):
            t = get_text(p)
            if 'button' in t.lower() or 'train' in t.lower() or 'categor' in t.lower() or 'stimul' in t.lower() or 'response' in t.lower():
                print(t[:500] + "..." if len(t) > 500 else t)
