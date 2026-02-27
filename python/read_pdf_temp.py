import sys
import fitz

doc = fitz.open(r'c:\Users\cityz\IllI\newer_all\decoding bayesian estimation.pdf')
text = ""
for i in range(min(5, len(doc))):
    text += doc[i].get_text() + "\n\n"

print(text)
