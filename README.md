# 🎬 CineMatch — MovieLens Deep Learning Recommender

**Every mind has a movie.**

A Deep Learning Recommendation Scoring System that predicts personalized movie ratings using MovieLens and an Artificial Neural Network (ANN).

## Project Links

- 🌐 **Live Demo:** [CineMatch](https://cinematch-dl.streamlit.app/)
- 📓 **Notebook:** [MovieLens_DL_project.ipynb](./MovieLens_DL_project.ipynb)
- 🎞️ **Presentation:** [CineMatch Presentation](./CineMatch_presentation.html)

## Dataset
- **100,836** ratings
- **610** users
- **9,742** movies

## Model
ANN with Embedding and Dense Layers, trained using Adam and MSE, with hyperparameter tuning and regularization.

## Results

| Model | Test MAE | Test RMSE |
|---|---|---|
| Mean Rating | 0.8306 | 1.0502 |
| Baseline ANN | 0.7196 | 0.9492 |
| **Final ANN** | **0.6743** | **0.8834** |

The Final ANN achieved the lowest prediction errors.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations & Future Work

The model currently supports users and movies seen during training. Future improvements include newer MovieLens data and support for new users through initial movie ratings.
