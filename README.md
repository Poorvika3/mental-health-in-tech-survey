# 🧠 Mental Health in Tech Survey — EDA & Interactive Dashboard

> Uncovering what actually drives treatment-seeking for mental health in the tech industry — and why "having a benefit" isn't the same as employees *knowing* it exists.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

This project analyzes the **2014 OSMI (Open Sourcing Mental Illness) Mental Health in Tech Survey** — 1,259 responses from tech industry employees on workplace attitudes toward mental health. It goes beyond a static notebook with a **bonus comparison against the 2016 OSMI survey** and a fully interactive **Streamlit dashboard**.

**The headline finding:** employees don't avoid treatment because support doesn't exist — they avoid it because they're *not sure* it exists. Across benefits, care options, and anonymity protections, employees who answered **"Don't know"** behaved almost identically to those who answered **"No."** Awareness, not policy design, is the real gap.

---

## ✨ Features

- 🧹 **Robust data cleaning** — fixed `Age` outliers (from -1726 to 99,999,999,999 in the raw data) and standardized 49 raw `Gender` spellings into 3 clean categories
- 📊 **18 fully-annotated EDA charts** — each with a clear insight, not just a plot
- 🔁 **Bonus 2014 vs 2016 comparison** — sourced from the official OSMI 2016 survey
- 🖥️ **Interactive Streamlit dashboard** — live filtering by gender, country, company size, and remote-work status, with hover-enabled Plotly charts
- 💡 **Business recommendations** — translated straight from the data into actionable, prioritized steps for employers

---

## 🔑 Key Insights

| Finding | Detail |
|---|---|
| 🧬 **Family history is the strongest predictor** | ~75% treatment-seeking rate with a family history vs ~35% without |
| 🔕 **"Don't know" ≈ "No"** | Employees unsure about benefits/anonymity behave like those with no support at all |
| 🗓️ **Leave policy clarity matters most** | A clear gradient links leave difficulty to fear of disclosing a condition |
| 🌍 **Sample skews US/UK, male, early-career** | Findings should be read with that context in mind |
| 📈 **Treatment-seeking rose 2014 → 2016** | ~50.6% → ~58.5%, but the awareness gap barely moved |

---

## 📂 Project Structure

```
MentalHealth_Survey/
├── .streamlit/
│   └── config.toml              # Dashboard theme (colors, font)
├── eda_outputs/                 # 19 saved PNG charts from the EDA script
├── venv/                        # Local virtual environment (not pushed to GitHub)
├── mental_health_eda.py         # EDA script (VS Code / cell-mode runnable)
├── Mental_Health_in_Tech_EDA.ipynb   # Full notebook version of the EDA
├── Mental_Health_Streamlit.py   # Streamlit dashboard app
├── README.md 
├── requirements.txt             # Python dependencies
├── survey.csv                   # 2014 OSMI survey data
└── survey_2016.csv              # 2016 OSMI survey data (bonus comparison)
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/abhi-1009/mental-health-in-tech-survey.git
cd mental-health-in-tech-survey

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

**Run the EDA script** (VS Code, cell-by-cell or top to bottom):
```bash
python mental_health_eda.py
```

**Launch the interactive dashboard:**
```bash
streamlit run Mental_Health_Streamlit.py
```
Then open **http://localhost:8501** in your browser.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit`

---

## 🌐 Live Demo

🔗 _Add your deployed Streamlit Community Cloud link here once deployed._

---

## 📊 Dataset Source

- 2014 Survey: [OSMI Mental Health in Tech Survey](https://osmihelp.org/research/)
- 2016 Survey: [Kaggle — Mental Health in Tech 2016](https://www.kaggle.com/osmi/mental-health-in-tech-2016)

---

## 👤 Author

**Abhijit Sinha** — [@abhi-1009](https://github.com/abhi-1009)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">🙏 Thanks for stopping by — feedback and stars ⭐ are always welcome!</p>
