import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mental Health in Tech Survey", layout="wide", page_icon="🧠")
sns.set_theme(style="whitegrid")

PALETTE = {"Yes": "#55A868", "No": "#C44E52", "Maybe": "#DD8452", "Don't know": "#DD8452", "Not sure": "#DD8452"}

st.markdown("""
<style>
.reco-card {
    background: #FFFFFF;
    border-left: 5px solid #4C72B0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.reco-card h4 { margin: 0 0 0.3rem 0; color: #2C3E50; }
.reco-card p { margin: 0; color: #444; font-size: 0.93rem; }
.kpi-card {
    background: #FFFFFF;
    border-left: 5px solid #4C72B0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 0.9rem;
}
.kpi-icon { font-size: 1.6rem; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #2C3E50; margin: 0.2rem 0; }
.kpi-label { font-size: 0.85rem; color: #666; }
.hero-banner {
    background: linear-gradient(135deg, #4C72B0, #6A8FC7 60%, #8FA9D6);
    color: white;
    padding: 1.6rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
}
.hero-banner h1 { margin: 0; font-size: 2rem; }
.hero-banner p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }
.thanks-box {
    text-align: center;
    padding: 1.4rem;
    margin-top: 1.5rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #EAF2FB, #F3EAF8);
    color: #2C3E50;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Data loading & cleaning (cached so it only runs once per session)
# ----------------------------------------------------------------------------
def clean_gender(g):
    g = str(g).strip().lower()
    male_terms = ['male', 'm', 'man', 'cis male', 'malr', 'maile', 'make', 'msle',
                  'mail', 'cis man', 'male-ish', 'male (cis)', 'guy (-ish) ^_^']
    female_terms = ['female', 'f', 'woman', 'cis female', 'femake', 'femail',
                     'female (trans)', 'cis-female/femme', 'trans-female',
                     'trans woman', 'female (cis)']
    if g in male_terms:
        return 'Male'
    elif g in female_terms:
        return 'Female'
    else:
        return 'Other'

@st.cache_data
def load_2014():
    df = pd.read_csv('survey.csv')
    df['Age'] = df['Age'].apply(lambda x: x if 18 <= x <= 75 else np.nan)
    df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
    df['Gender'] = df['Gender'].apply(clean_gender)
    df['state'] = df['state'].fillna('Not Applicable')
    df['self_employed'] = df['self_employed'].fillna('No')
    df['work_interfere'] = df['work_interfere'].fillna('Not applicable')
    df = df.drop(columns=['comments'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
    df['no_employees'] = pd.Categorical(df['no_employees'], categories=size_order, ordered=True)

    df_encoded = df.copy()
    binary_map = {'Yes': 1, 'No': 0}
    for c in ['treatment', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']:
        df_encoded[c] = df_encoded[c].map(binary_map)
    return df, df_encoded

@st.cache_data
def load_2016():
    df16 = pd.read_csv('survey_2016.csv')
    rename_map = {
        'What is your age?': 'Age',
        'What is your gender?': 'Gender',
        'What country do you live in?': 'Country',
        'Are you self-employed?': 'self_employed',
        'How many employees does your company or organization have?': 'no_employees',
        'Do you have a family history of mental illness?': 'family_history',
        'Have you ever sought treatment for a mental health issue from a mental health professional?': 'treatment',
        'Does your employer provide mental health benefits as part of healthcare coverage?': 'benefits',
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': 'anonymity',
        'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:': 'leave',
        'Do you work remotely?': 'remote_work',
    }
    df16 = df16.rename(columns=rename_map)
    df16['Age'] = df16['Age'].apply(lambda x: x if 18 <= x <= 75 else np.nan)
    df16['Age'] = df16['Age'].fillna(df16['Age'].median()).astype(int)
    df16['Gender'] = df16['Gender'].apply(clean_gender)
    df16['treatment'] = df16['treatment'].map({1: 'Yes', 0: 'No'})
    df16['remote_work'] = df16['remote_work'].map({'Always': 'Yes', 'Sometimes': 'Yes', 'Never': 'No'})
    for c in ['benefits', 'anonymity']:
        df16[c] = df16[c].replace({"I don't know": "Don't know"})
    return df16

def pct_stacked_bar(data, index_col, value_col, order=None, value_order=('Yes', 'No'), title=""):
    """Build a % stacked bar chart (Plotly) with hover tooltips from a raw dataframe."""
    ct = pd.crosstab(data[index_col], data[value_col], normalize='index') * 100
    ct = ct.reindex(columns=[v for v in value_order if v in ct.columns])
    if order is not None:
        ct = ct.reindex(order)
    long_df = ct.reset_index().melt(id_vars=index_col, var_name=value_col, value_name='pct')
    fig = px.bar(long_df, x=index_col, y='pct', color=value_col, barmode='stack',
                 color_discrete_map=PALETTE, title=title,
                 labels={'pct': '% of Respondents', index_col: index_col})
    fig.update_traces(hovertemplate='%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>')
    fig.update_layout(legend_title_text=value_col, height=400, margin=dict(t=50, b=10))
    return fig

df, df_encoded = load_2014()
df16 = load_2016()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='text-align:center; font-size:2.4rem; margin-bottom:-0.5rem;'>🧠</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<h3 style='text-align:center; margin-top:0;'>Survey Explorer</h3>",
    unsafe_allow_html=True
)
st.sidebar.divider()

size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
top_countries_all = df['Country'].value_counts().head(15).index.tolist()

with st.sidebar.expander("👤 Demographics", expanded=True):
    gender_sel = st.multiselect("Gender", options=sorted(df['Gender'].unique()),
                                 default=sorted(df['Gender'].unique()))
    country_sel = st.multiselect("Country (top 15 shown, blank = all)",
                                  options=top_countries_all, default=[])

with st.sidebar.expander("🏢 Workplace", expanded=True):
    size_sel = st.multiselect("Company size", options=size_order, default=size_order)
    remote_sel = st.radio("Remote work", options=["All", "Yes", "No"], horizontal=True)

mask = df['Gender'].isin(gender_sel) & df['no_employees'].isin(size_sel)
if country_sel:
    mask &= df['Country'].isin(country_sel)
if remote_sel != "All":
    mask &= df['remote_work'] == remote_sel
fdf = df[mask].copy()

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(fdf)}** of {len(df)} respondents based on current filters.")
st.sidebar.download_button(
    "⬇️ Download filtered data (CSV)",
    data=fdf.to_csv(index=False).encode('utf-8'),
    file_name="filtered_survey_data.csv",
    mime="text/csv",
    use_container_width=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🧠 Mental Health in Tech Survey</h1>
    <p>2014 OSMI Survey · 1,259 tech-industry respondents · with a bonus 2016 comparison</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_charts, tab_2016, tab_reco = st.tabs(
    ["📋 Overview", "📊 EDA Charts", "🔁 2014 vs 2016", "💡 Recommendations"]
)

# ----------------------------------------------------------------------------
# Tab 1: Overview
# ----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Project Summary")
    st.write(
        "This project explores the 2014 OSMI survey on attitudes toward mental health and the "
        "frequency of mental health conditions in the tech workplace. After cleaning invalid `Age` "
        "values and standardizing 49 raw `Gender` spellings, the analysis looks at who seeks "
        "treatment and which employer-side factors (benefits, care options, anonymity, leave policy) "
        "are associated with it."
    )

    st.subheader("Problem Statement")
    st.write(
        "Mental health remains under-addressed in tech workplaces relative to physical health. "
        "Employers often don't know whether existing policies (benefits, wellness programs, leave) "
        "are actually known about or trusted by employees, and employees may stay silent for fear "
        "of professional consequences."
    )

    st.subheader("Business Objective")
    st.write(
        "Identify which employer policies and workplace characteristics are most strongly associated "
        "with (a) seeking treatment and (b) willingness to discuss mental health openly - so employers "
        "can prioritize the interventions most likely to reduce stigma and increase treatment-seeking."
    )

    st.divider()
    st.subheader("Key Metrics (filtered)")

    treat_rate = (fdf['treatment'] == 'Yes').mean() * 100 if len(fdf) else 0
    fam_rate = (fdf['family_history'] == 'Yes').mean() * 100 if len(fdf) else 0
    anon_dk = (fdf['anonymity'] == "Don't know").mean() * 100 if len(fdf) else 0

    kpis = [
        ("👥", "Respondents", f"{len(fdf)}"),
        ("💊", "Sought Treatment", f"{treat_rate:.1f}%"),
        ("🧬", "Family History", f"{fam_rate:.1f}%"),
        ("🔒", "Unsure re: Anonymity", f"{anon_dk:.1f}%"),
    ]
    k1, k2, k3, k4 = st.columns(4)
    for col, (icon, label, value) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    gauge_col, note_col = st.columns([1, 1])
    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=treat_rate,
            number={'suffix': "%"},
            title={'text': "Treatment-Seeking Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4C72B0"},
                'steps': [
                    {'range': [0, 40], 'color': "#F4CCCC"},
                    {'range': [40, 70], 'color': "#FCE8B2"},
                    {'range': [70, 100], 'color': "#D9EAD3"},
                ],
            },
        ))
        fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    with note_col:
        st.markdown(
            "<div style='padding-top:2.2rem; color:#555; font-size:0.95rem;'>"
            "This gauge tracks the same 'Sought Treatment' figure as the card on the left, just "
            "at a glance - green (70-100%) means most respondents in the current filter have sought "
            "treatment; red (0-40%) flags a group where most haven't."
            "</div>", unsafe_allow_html=True
        )

    st.subheader("Filtered Data Preview")
    st.dataframe(fdf.head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: EDA Charts
# ----------------------------------------------------------------------------
with tab_charts:
    if len(fdf) < 5:
        st.warning("Too few respondents match the current filters to chart meaningfully. Widen the filters.")
    else:
        colA, colB = st.columns(2)
        with colA:
            fig = px.histogram(fdf, x='Age', nbins=20, title="Age Distribution",
                                color_discrete_sequence=['#4C72B0'])
            fig.update_layout(height=380, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            gcounts = fdf['Gender'].value_counts().reset_index()
            gcounts.columns = ['Gender', 'Count']
            fig = px.bar(gcounts, x='Gender', y='Count', title="Gender Distribution",
                         color='Gender', color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=380, margin=dict(t=50, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            fig = pct_stacked_bar(fdf, 'family_history', 'treatment',
                                   title="Treatment Rate by Family History")
            st.plotly_chart(fig, use_container_width=True)
        with colD:
            fig = pct_stacked_bar(fdf, 'benefits', 'treatment', order=['Yes', "Don't know", 'No'],
                                   title="Treatment Rate by Benefits Awareness")
            st.plotly_chart(fig, use_container_width=True)

        colE, colF = st.columns(2)
        with colE:
            fig = pct_stacked_bar(fdf, 'anonymity', 'treatment', order=['Yes', "Don't know", 'No'],
                                   title="Treatment Rate by Anonymity Protection")
            st.plotly_chart(fig, use_container_width=True)
        with colF:
            leave_order = ['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult']
            fig = pct_stacked_bar(fdf, 'leave', 'mental_health_consequence', order=leave_order,
                                   value_order=('No', 'Maybe', 'Yes'),
                                   title="Negative Consequence by Ease of Leave")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Correlation Heatmap** (encoded key variables, filtered subset)")
        corr_cols = ['Age', 'treatment', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']
        f_encoded = df_encoded.loc[fdf.index]
        corr = f_encoded[corr_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=450, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📁 See all 18 pre-rendered EDA charts from the notebook (static images)"):
            st.caption(
                "These come from the EDA script's `eda_outputs/` folder. They're static PNGs, "
                "so they won't respond to hover like the charts above - the interactive versions "
                "are what you see on this tab."
            )
            import os
            chart_files = sorted(f for f in os.listdir('eda_outputs') if f.endswith('.png')) \
                if os.path.isdir('eda_outputs') else []
            if not chart_files:
                st.info("Run the EDA script first so `eda_outputs/` has the saved chart images.")
            else:
                cols = st.columns(3)
                for i, fname in enumerate(chart_files):
                    with cols[i % 3]:
                        st.image(os.path.join('eda_outputs', fname), caption=fname, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 3: 2014 vs 2016 comparison (static - not affected by sidebar filters)
# ----------------------------------------------------------------------------
with tab_2016:
    st.caption(
        "The 2016 survey asks a different question set (63 columns, full-sentence headers), so it's "
        "compared separately here rather than merged with the 2014 data above."
    )

    rate_2014 = (df['treatment'] == 'Yes').mean() * 100
    rate_2016 = (df16['treatment'] == 'Yes').mean() * 100
    c1, c2 = st.columns(2)
    c1.metric("2014 Treatment Rate", f"{rate_2014:.1f}%")
    c2.metric("2016 Treatment Rate", f"{rate_2016:.1f}%", delta=f"{rate_2016 - rate_2014:.1f} pts")

    colA, colB = st.columns(2)
    with colA:
        rows = []
        for col, label in [('benefits', 'Benefits'), ('anonymity', 'Anonymity')]:
            for year, data in [('2014', df), ('2016', df16)]:
                vc = data[col].value_counts(normalize=True).reindex(['Yes', "Don't know", 'No']) * 100
                for cat, val in vc.items():
                    rows.append({'Question': label, 'Response': cat, 'Year': year, 'pct': val})
        comp_df = pd.DataFrame(rows)
        fig = px.bar(comp_df, x='Response', y='pct', color='Year', barmode='group',
                     facet_col='Question', title="Benefits & Anonymity Awareness",
                     color_discrete_map={'2014': '#4C72B0', '2016': '#DD8452'},
                     labels={'pct': '% of Respondents'})
        fig.update_layout(height=420, margin=dict(t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        rows = []
        for year, data in [('2014', df), ('2016', df16)]:
            vc = data['no_employees'].value_counts(normalize=True).reindex(size_order) * 100
            for cat, val in vc.items():
                rows.append({'Company Size': cat, 'Year': year, 'pct': val})
        comp_df = pd.DataFrame(rows)
        fig = px.bar(comp_df, x='Company Size', y='pct', color='Year', barmode='group',
                     title="Company Size Distribution",
                     color_discrete_map={'2014': '#4C72B0', '2016': '#DD8452'},
                     labels={'pct': '% of Respondents'})
        fig.update_layout(height=420, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The treatment-seeking rate rose from 2014 to 2016, but the 'Don't know' share for benefits "
        "and anonymity stayed large in both years - the awareness gap persisted even as overall "
        "openness improved."
    )

# ----------------------------------------------------------------------------
# Tab 4: Recommendations & Conclusion
# ----------------------------------------------------------------------------
with tab_reco:
    st.subheader("Solution to the Business Objective")

    recommendations = [
        ("1️⃣", "Make benefits & care options unmissable",
         "Communicate them at onboarding and annually - \"Don't know\" behaves like \"No\" in practice."),
        ("2️⃣", "Clarify medical leave policy",
         "The strongest gradient in the data links leave difficulty to fear of disclosure."),
        ("3️⃣", "Proactively communicate anonymity protections",
         "This had the largest uncertainty gap of all the awareness variables (~65% unsure)."),
        ("4️⃣", "Don't over-index on remote work or company size",
         "Both showed weak relationships with treatment-seeking in this data."),
        ("5️⃣", "Use family history to shape outreach, not gatekeep it",
         "It's the single strongest predictor found - useful for opt-in EAP intake design."),
    ]
    rc1, rc2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(recommendations):
        target = rc1 if i % 2 == 0 else rc2
        with target:
            st.markdown(f"""
            <div class="reco-card">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Conclusion")
    st.write(
        "Whether an employee seeks treatment is driven less by demographics or company structure and "
        "more by family history and how clearly an employer communicates its mental health support. "
        "The clearest, most actionable pattern is the link between leave-policy clarity and fear of "
        "disclosure - a low-cost policy fix with a plausibly large effect on outcomes."
    )

    # Visual summary: effect size of each key driver, computed from the full 2014 dataset
    fh_rates = df.groupby('family_history')['treatment'].apply(lambda s: (s == 'Yes').mean() * 100)
    fh_gap = fh_rates.get('Yes', 0) - fh_rates.get('No', 0)

    ben_rates = df.groupby('benefits')['treatment'].apply(lambda s: (s == 'Yes').mean() * 100)
    ben_gap = ben_rates.get('Yes', 0) - ben_rates.get('No', 0)

    anon_rates = df.groupby('anonymity')['treatment'].apply(lambda s: (s == 'Yes').mean() * 100)
    anon_gap = anon_rates.get('Yes', 0) - anon_rates.get("Don't know", 0)

    leave_conseq = df.groupby('leave')['mental_health_consequence'].apply(lambda s: (s == 'Yes').mean() * 100)
    leave_gap = leave_conseq.max() - leave_conseq.min()

    driver_df = pd.DataFrame({
        'Driver': ['Family history\n(treatment gap)', 'Benefits awareness\n(treatment gap)',
                   'Anonymity awareness\n(treatment gap)', 'Leave difficulty\n(consequence-fear gap)'],
        'Effect (percentage points)': [fh_gap, ben_gap, anon_gap, leave_gap]
    }).sort_values('Effect (percentage points)', ascending=True)

    fig = px.bar(driver_df, x='Effect (percentage points)', y='Driver', orientation='h',
                 title="Key Drivers - Size of Effect (percentage-point gap)",
                 color='Effect (percentage points)', color_continuous_scale='Blues')
    fig.update_layout(height=350, margin=dict(t=50, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Family history shows the largest gap in who seeks treatment; leave-policy clarity shows the "
        "largest gap in who fears negative consequences from disclosing - together, the two strongest "
        "levers found in this analysis."
    )

    st.markdown("""
    <div class="thanks-box">
        <h3>🙏 Thank You</h3>
        <p>Thank you for taking the time to review this Mental Health in Tech Survey project.<br>
        Feedback and suggestions are always welcome!</p>
    </div>
    """, unsafe_allow_html=True)