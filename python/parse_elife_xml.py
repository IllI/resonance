import xml.etree.ElementTree as ET

tree = ET.parse('elife.xml')
root = tree.getroot()

def get_text(element):
    return "".join(element.itertext())

with open("elife_methods_extracted.txt", "w", encoding="utf-8") as f:
    for p in root.iter('p'):
        t = get_text(p).lower()
        if 'button' in t or 'press' in t or 'stimuli' in t or 'category' in t or 'train' in t:
            f.write(get_text(p) + "\n\n")
