# Crypto Financial Intelligence Platform (Group 11)

Streamlit web app for the 36120 Advanced Machine Learning Application AT3 project. Users choose one of eight
cryptocurrencies and see 7-day machine-learning forecasts from four independent FastAPI services, one per student.
No model runs inside Streamlit: every prediction, feature and piece of model metadata comes from the APIs over HTTP.

## Group 11

| Name | Student ID | Role | Page |
|---|---|---|---|
| Vivek Rai | 25731011 | Student A | Asset Performance Index (API) |
| Sai Lokesh Arvapalli | 25598311 | lokeshUTS | Student B | Market Risk Index
| Sai Prasad Kammari | 25693969 | saiprasad0722 | Student C | Trend Category (TC) |
| Arsh Mansuri | 25522436 | Student D | Trading Opportunity Rating (TOR) |

## Live services

| Component | URL |
|---|---|
| Streamlit app | [add after deployment] |
| Student A API | https://group11-studenta-api.onrender.com |
| Student B API | https://group11-studentb-api.onrender.com |
| Student C API | https://group11-studentc-api.onrender.com |
| Student D API | [add URL] |

The APIs run on Render's free plan and sleep after about 15 minutes without traffic. Open each `/health` page a
minute before using the app so every service is awake.

## Project structure

```
app/
  main.py            Home page, top navigation, coin picker and the shared page layout
students/
  ui.py              Shared design kit: colours, banners, cards, gauges, charts
  student_a.py       Asset Performance Index page (Student A)
  student_b.py       Market Risk Index page (Student B)
  student_c.py       Trend Category page (Student C)
  student_d.py       Trading Opportunity Rating page (Student D)
.streamlit/
  config.toml        App theme
requirements.txt     Pinned dependencies
pyproject.toml       Project metadata
Dockerfile           Container build for running the app anywhere
github.txt           Repository URL
```

Each `students/student_x.py` provides two functions that `app/main.py` calls:

- `render(token)`: the student's full page
- `home_chart(token)`: one compact visual for the home page

`app/main.py` calls each of them inside its own error handler, so if one student's API or page fails, the rest of
the app keeps working and shows a clear message for that section.

## Run locally

Requires Python 3.12.13.

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

Then open http://localhost:8501

### Pointing the app at different APIs

Each page reads its API URL from an environment variable and falls back to the deployed Render URL:

| Variable | Default |
|---|---|
| `STUDENT_A_API_URL` | https://group11-studenta-api.onrender.com |
| `STUDENT_B_API_URL` | https://group11-studentb-api.onrender.com |

## Run with Docker

```bash
docker build -t group11-streamlit .
docker run -p 8501:8501 group11-streamlit
```

Then open http://localhost:8501

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with GitHub and click **Create app**.
2. Repository: `36120-26SP-AT3-Group11/36120-26SP-AT3-Group11-streamlit`, branch `main`, main file `app/main.py`.
3. Under **Advanced settings**, choose Python 3.12.
4. Click **Deploy**.

## Contributing

`main` is protected: work on your own branch, open a pull request, and merge after one teammate approves.
