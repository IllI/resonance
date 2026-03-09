import os
from nilearn import datasets

print("Verifying Anatomical Scans for Haxby Subjects 1-6...\n")

# Fetch all 6 subjects
# fetch_kwargs={'resume': True}
try:
    haxby_data = datasets.fetch_haxby(subjects=[1, 2, 3, 4, 5, 6], data_dir=r"C:\Users\cityz\IllI\newer_all\data\nilearn_data")
    
    # Check what is returned
    for i, subj_num in enumerate([1, 2, 3, 4, 5, 6]):
        print(f"--- Subject {subj_num} ---")
        
        # Check functional data
        if i < len(haxby_data.func):
            print(f"  Functional Data: Found ({haxby_data.func[i]})")
        else:
            print(f"  Functional Data: MISSING")
            
        # Check anatomical data
        if i < len(haxby_data.anat):
            anat_file = haxby_data.anat[i]
            if str(anat_file).lower() == 'none' or not anat_file or not os.path.exists(anat_file):
                print(f"  Anatomical Data: MISSING (Returned: {anat_file})")
            else:
                print(f"  Anatomical Data: Found ({anat_file})")
        else:
            print(f"  Anatomical Data: MISSING (Not provided in dataset structure)")

except Exception as e:
    print(f"Error fetching data: {e}")
