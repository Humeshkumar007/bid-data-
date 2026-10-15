
import os
import urllib.request
import zipfile


def download_and_extract():
    os.makedirs('data/raw', exist_ok=True)
    zip_path = 'data/anonymisedData.zip'
    url = 'https://ndownloader.figshare.com/files/8606371'
    
    if not os.path.exists(zip_path):
        print(f"Downloading OULAD dataset from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists.")
        
    print("Extracting files to data/raw ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/raw')
    
    print("\nExtraction complete! Files in data/raw:")
    for root, dirs, files in os.walk('data/raw'):
        for f in files:
            fp = os.path.join(root, f)
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  - {f}: {size_mb:.2f} MB")

if __name__ == '__main__':
    download_and_extract()
