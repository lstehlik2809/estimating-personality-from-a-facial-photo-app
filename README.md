# Estimating personality from a facial photo: web app

An interactive, in-browser version of the experimental Python script from
[Estimating-personality-from-a-facial-photo](https://github.com/lstehlik2809/Estimating-personality-from-a-facial-photo).

Upload a photo or take one with your camera. The app compares the face with composite faces that average
people high and low in each Big Five trait, from
[Little & Perrett (2007)](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1348/000712606X109648), and reports
which composite of each pair the face is closer to, together with the absolute difference in cosine similarity.

**Caveat: This app is intended strictly for experimental and educational purposes. It does not provide a valid,
reliable, or scientifically endorsed method for inferring personality traits from facial images. Personality is a
complex, multi-dimensional construct that cannot be accurately assessed from appearance alone. Any apparent
patterns or outputs should be interpreted as illustrative artifacts of the dataset and method, not as diagnostic
or predictive insights. Use responsibly and with critical awareness of ethical and scientific limitations.**

## How it works

Everything runs in the browser; the photo never leaves the device.

1. A face detector (MediaPipe BlazeFace) finds the face, the image is rotated so the eyes are level, and the face is cropped.
2. FaceNet-512, the model the original script uses through DeepFace, converted to ONNX and run with ONNX Runtime Web, turns the crop into a 512-number embedding.
3. The composite faces go through the same steps. For each trait, cosine similarity to the "high" and "low" composite is compared.

## Differences from the Python script

- The script uses DeepFace's default OpenCV face detector; the app uses MediaPipe, with a crop sized to match
  the OpenCV boxes. Similarities therefore differ a little from the script's output. On the sample photo from the original repo, all
  five verdicts are the same and the similarity gaps are of similar size.
- Model weights are stored in half precision to halve the download (see `tools/convert_model.py`).

## Run locally

```bash
python -m http.server 8765
```

Then open http://localhost:8765. Camera capture needs `localhost` or HTTPS.

## Files

- `index.html`, `css/`, `js/`: the app
- `images/`: composite faces from the original repo
- `models/`: FaceNet-512 (ONNX) and the face detector
- `tools/convert_model.py`: recreates `models/facenet512.onnx` from DeepFace's weights
