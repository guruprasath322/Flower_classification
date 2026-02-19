# Flower Classification App 🌸

A Streamlit web application for flower classification using deep learning (TensorFlow/Keras).

## Features
- Upload flower images (JPG, JPEG, PNG)
- Real-time flower classification
- Supports 5 flower types: Daisy, Dandelion, Rose, Sunflower, Tulip
- Shows prediction confidence

## Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/guruprasath322/Flower_classification.git
cd Flower_classification
```
📂 Dataset Link

Download the dataset here:
➡ https://drive.google.com/drive/folders/1rzEFpraXXLLIGPo6F6jNEpeMOdWMgOzN

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model (if needed)**
```bash
python train_model.py
```

5. **Run the app**
```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## Streamlit Cloud Deployment

### Prerequisites:
- GitHub account with repository pushed
- Streamlit account (free at [share.streamlit.io](https://share.streamlit.io))
- Model file `flower_model.keras` in repository

### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository and set `app.py` as main file
   - Click Deploy

3. **Your app will be live at**:
   `https://<username>-flower-classification.streamlit.app`

## Model Information
- **Architecture**: CNN (Convolutional Neural Network)
- **Input Size**: 150x150 RGB images
- **Classes**: 5 flower types
- **Accuracy**: ~65% validation accuracy

## Troubleshooting

- **Model not found**: Ensure `flower_model.keras` is in the project root
- **Large file issues**: Use Git LFS for files > 100MB
- **Authentication errors**: Use personal access token for GitHub

## License
MIT
