import gdown
import yaml
import hashlib

def download_model():
    data = None
    with open("data/model.yaml", "rt") as file:
        data = yaml.safe_load(file)

    md5 = None
    try:
        with open(file="models/" + data["filename"], mode="rb") as file:
            print("Checking if model is downloaded...")
            md5 = hashlib.file_digest(file, "md5").hexdigest()
            if md5 == data["md5hash"]:
                print("Model is aldready downloaded!")
                return
    except:
        pass

    print("Downloading model")
    gdown.download(
        id="1dJnjI1SmIFhq6VnBuNxXwPdup71yem34",
        output="models/" + data["filename"],
        fuzzy=True,
    )

if __name__ == "__main__":
    download_model()
