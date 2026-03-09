import urllib.request
import traceback

with open("doc_colab_text.txt", "w", encoding="utf-8") as f:
    try:
        doc_url = "https://docs.google.com/document/d/1mZzgz7C_R4kewDWzKwi-aLm9-3jrW4ZVs1sSB0e76jg/export?format=txt"
        doc_text = urllib.request.urlopen(doc_url).read().decode('utf-8')
        f.write("--- DOC ---\n")
        f.write(doc_text)
    except Exception as e:
        f.write(f"Doc Fetch Error: {e}\n{traceback.format_exc()}\n")

    f.write("\n\n--- COLAB ---\n")
    try:
        colab_url = "https://drive.google.com/uc?id=13nRLSpev5aMSZjwOONTcyuEeExLbRYoR&export=download"
        colab_text = urllib.request.urlopen(colab_url).read().decode('utf-8')
        f.write("Colab Content Length: " + str(len(colab_text)) + "\n")
        f.write(colab_text[:5000])
    except Exception as e:
        f.write(f"Colab Fetch Error: {e}\n{traceback.format_exc()}\n")
