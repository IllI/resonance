
try:
    import openneuro
    print("Has openneuro")
    print(dir(openneuro))
    if hasattr(openneuro, 'download'):
        print("openneuro.download found")
except Exception as e:
    print(e)
