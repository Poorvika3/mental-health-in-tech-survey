# # Mental Health in Tech Survey - EDA

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (9, 5)

import os
os.makedirs('eda_outputs', exist_ok=True)

# Load dataset
df = pd.read_csv('survey.csv')
df.head()

# Shape, info, duplicates, missing values
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
df.info()
print(f"Duplicate rows: {df.duplicated().sum()}")
df.isnull().sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.savefig('eda_outputs/missing_values_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Columns / describe / unique values
df.columns.tolist()
df.describe(include='all').T

for col in df.columns:
    print(f"{col} ({df[col].nunique()} unique): {df[col].unique()[:8]}")
    print("-" * 80)

# ## Data Wrangling
# - Age: capped to 18-75, rest imputed with median (raw data had values from -1726 to 99,999,999,999)
# - Gender: 49 raw spellings standardized to Male / Female / Other
# - Missing values filled based on why they're missing (state -> 'Not Applicable', self_employed -> 'No',
#   work_interfere -> 'Not applicable' since it's mostly non-treatment-seekers)
# - comments dropped (free text, not used here); Timestamp parsed to datetime
# - no_employees ordered as a category for consistent chart ordering
# - df_encoded created with Yes/No -> 1/0 for the correlation heatmap & pair plot later

# 1. Clean Age
df['Age'] = df['Age'].apply(lambda x: x if 18 <= x <= 75 else np.nan)
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)


# 2. Standardize Gender
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

df['Gender'] = df['Gender'].apply(clean_gender)

# 3. Handle missing values
df['state'] = df['state'].fillna('Not Applicable')
df['self_employed'] = df['self_employed'].fillna('No')
df['work_interfere'] = df['work_interfere'].fillna('Not applicable')

# 4. Drop comments column
df.drop(columns=['comments'], inplace=True)

# 5. Parse Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 6. Order company-size buckets
size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
df['no_employees'] = pd.Categorical(df['no_employees'], categories=size_order, ordered=True)

# 7. Encoded copy for correlation analysis
binary_map = {'Yes': 1, 'No': 0}
df_encoded = df.copy()
for c in ['treatment', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']:
    df_encoded[c] = df_encoded[c].map(binary_map)
df.head()

## Charts

# ### Chart 1 - Age Distribution
# **Insight:** Centered late-20s to mid-30s, right-skewed - a mostly early/mid-career workforce.

plt.figure(figsize=(9, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='#4C72B0')
plt.title('Distribution of Respondent Age (Cleaned)')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('eda_outputs/01_age_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 2 - Gender Distribution
# **Insight:** ~79% Male, Female next, Other a small minority - gender-split findings elsewhere carry wider uncertainty for the Female/Other groups.

plt.figure(figsize=(7, 5))
order = df['Gender'].value_counts().index
sns.countplot(data=df, x='Gender', order=order, hue='Gender', palette='Set2', legend=False)
plt.title('Respondent Gender Distribution (Standardized)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('eda_outputs/02_gender_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 3 - Treatment Sought (target variable)
# **Insight:** ~51% have sought treatment, ~49% haven't - a near-even split worth explaining.

plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='treatment', order=['Yes', 'No'], hue='treatment',
              palette=['#55A868', '#C44E52'], legend=False)
plt.title('Have Respondents Sought Treatment for a Mental Health Condition?')
plt.xlabel('Sought Treatment')
plt.ylabel('Count')
plt.savefig('eda_outputs/03_treatment_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 4 - Family History vs Treatment
# **Insight:** Family history -> ~75% treatment rate vs ~35% without. One of the strongest single predictors of treatment-seeking in the dataset.

ct = pd.crosstab(df['family_history'], df['treatment'], normalize='index') * 100
ct = ct[['Yes', 'No']]
ct.plot(kind='bar', stacked=True, color=['#55A868', '#C44E52'], figsize=(7, 5))
plt.title('Treatment-Seeking Rate by Family History of Mental Illness')
plt.xlabel('Family History of Mental Illness')
plt.ylabel('% of Respondents')
plt.xticks(rotation=0)
plt.legend(title='Sought Treatment', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/04_family_history_vs_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 5 - Work Interference Distribution
# **Insight:** Excluding "Not applicable", "Sometimes" is most common - work interference is closer to the norm than the exception for those with a condition.

order = ['Never', 'Rarely', 'Sometimes', 'Often', 'Not applicable']
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='work_interfere', order=order, hue='work_interfere',
              palette='mako', legend=False)
plt.title('How Often Mental Health Interferes With Work')
plt.xlabel('Work Interference')
plt.ylabel('Count')
plt.savefig('eda_outputs/05_work_interfere_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 6 - Top 10 Countries
# **Insight:** US (~60%) and UK dominate the sample - country-specific conclusions elsewhere are only reliable for these two.

top_countries = df['Country'].value_counts().head(10)
plt.figure(figsize=(9, 5))
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index,
            palette='crest', legend=False)
plt.title('Top 10 Countries by Number of Respondents')
plt.xlabel('Number of Respondents')
plt.ylabel('Country')
plt.savefig('eda_outputs/06_top10_countries.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 7 - Company Size Distribution
# **Insight:** Fairly spread across sizes, concentrated at "More than 1000" and "26-100" - findings hold across both startup and enterprise contexts.

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='no_employees',
              order=['1-5','6-25','26-100','100-500','500-1000','More than 1000'],
              hue='no_employees', palette='flare', legend=False)
plt.title('Respondent Distribution by Company Size')
plt.xlabel('Number of Employees')
plt.ylabel('Count')
plt.xticks(rotation=20)
plt.savefig('eda_outputs/07_company_size_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 8 - Remote Work vs Treatment
# **Insight:** Treatment-seeking rates are nearly identical for remote vs non-remote - remote status alone isn't a meaningful driver.

ct = pd.crosstab(df['remote_work'], df['treatment'], normalize='index') * 100
ct = ct[['Yes', 'No']]
ct.plot(kind='bar', stacked=True, color=['#55A868', '#C44E52'], figsize=(7, 5))
plt.title('Treatment-Seeking Rate by Remote Work Status')
plt.xlabel('Works Remotely (>= 50% of time)')
plt.ylabel('% of Respondents')
plt.xticks(rotation=0)
plt.legend(title='Sought Treatment', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/08_remote_work_vs_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 9 - Benefits vs Treatment
# **Insight:** Knowing benefits exist -> higher treatment rate; "Don't know" behaves like "No", not like a neutral middle option.

ct = pd.crosstab(df['benefits'], df['treatment'], normalize='index') * 100
ct = ct[['Yes', 'No']]
ct = ct.reindex(['Yes', "Don't know", 'No'])
ct.plot(kind='bar', stacked=True, color=['#55A868', '#C44E52'], figsize=(7, 5))
plt.title('Treatment-Seeking Rate by Employer-Provided Mental Health Benefits')
plt.xlabel('Employer Provides Mental Health Benefits')
plt.ylabel('% of Respondents')
plt.xticks(rotation=0)
plt.legend(title='Sought Treatment', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/09_benefits_vs_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 10 - Care Options vs Treatment
# **Insight:** Same pattern as benefits - "Not sure" tracks closer to "No" than "Yes", reinforcing that awareness (not just existence) of support matters.

ct = pd.crosstab(df['care_options'], df['treatment'], normalize='index') * 100
ct = ct[['Yes', 'No']]
ct = ct.reindex(['Yes', 'Not sure', 'No'])
ct.plot(kind='bar', stacked=True, color=['#55A868', '#C44E52'], figsize=(7, 5))
plt.title('Treatment-Seeking Rate by Awareness of Care Options')
plt.xlabel('Knows Employer-Provided Care Options')
plt.ylabel('% of Respondents')
plt.xticks(rotation=0)
plt.legend(title='Sought Treatment', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/10_care_options_vs_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 11 - Wellness Program vs Treatment
# **Insight:** Weaker gap than benefits/care_options - wellness-program messaging is a lower-priority lever than clear benefits communication.

ct = pd.crosstab(df['wellness_program'], df['treatment'], normalize='index') * 100
ct = ct[['Yes', 'No']]
ct = ct.reindex(['Yes', "Don't know", 'No'])
ct.plot(kind='bar', stacked=True, color=['#55A868', '#C44E52'], figsize=(7, 5))
plt.title('Treatment-Seeking Rate by Wellness Program Discussion')
plt.xlabel('Employer Discussed Mental Health in Wellness Program')
plt.ylabel('% of Respondents')
plt.xticks(rotation=0)
plt.legend(title='Sought Treatment', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/11_wellness_program_vs_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 12 - Anonymity vs Treatment
# **Insight:** Largest "Don't know" group of the three awareness variables, and it sits well below "Yes" on treatment-seeking - confidentiality uncertainty is a real deterrent.

ct = pd.crosstab(df['anonymity'], df['treatment'], normalize='index') * 100
ct = ct[['Yes', 'No']]
ct = ct.reindex(['Yes', "Don't know", 'No'])
ct.plot(kind='bar', stacked=True, color=['#55A868', '#C44E52'], figsize=(7, 5))
plt.title('Treatment-Seeking Rate by Anonymity Protection')
plt.xlabel('Anonymity Protected for Treatment Resources')
plt.ylabel('% of Respondents')
plt.xticks(rotation=0)
plt.legend(title='Sought Treatment', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/12_anonymity_vs_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 13 - Leave Ease vs Negative Consequence
# **Insight:** Clear gradient - harder/unclear leave policy tracks with a higher expectation of negative consequences from disclosing. One of the strongest signals in the dataset.

leave_order = ['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult']
ct = pd.crosstab(df['leave'], df['mental_health_consequence'], normalize='index') * 100
ct = ct.reindex(leave_order)[['No', 'Maybe', 'Yes']]
ct.plot(kind='bar', stacked=True, color=['#55A868', '#DD8452', '#C44E52'], figsize=(8, 5))
plt.title('Perceived Negative Consequences by Ease of Taking Mental Health Leave')
plt.xlabel('Ease of Taking Medical Leave for Mental Health')
plt.ylabel('% Expecting Negative Consequences if Disclosed')
plt.xticks(rotation=20)
plt.legend(title='Expects Negative Consequence', bbox_to_anchor=(1.02, 1))
plt.savefig('eda_outputs/13_leave_vs_consequence.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 14 - Correlation Heatmap
# **Insight:** family_history <-> treatment is by far the strongest correlation; remote_work, tech_company, obs_consequence, Age are only weakly related to treatment.

corr_cols = ['Age', 'treatment', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']
corr = df_encoded[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap of Encoded Key Variables')
plt.savefig('eda_outputs/14_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 15 - Pair Plot
# **Insight:** Treatment Yes/No doesn't separate cleanly on Age, but does on family_history - confirming it as the most distinct driver found in this analysis.

pair_df = df_encoded[['Age', 'family_history', 'treatment']].copy()
pair_df['treatment'] = pair_df['treatment'].map({1: 'Yes', 0: 'No'})

g = sns.pairplot(pair_df, hue='treatment', palette=['#C44E52', '#55A868'], diag_kind='kde')
g.fig.suptitle('Pair Plot: Age & Family History by Treatment Status', y=1.02)
g.savefig('eda_outputs/15_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# ## Bonus - 2014 vs 2016 Comparison
# 2016 survey has a different question set (63 cols, full-sentence headers) so it's compared separately rather than merged. Source: OSMI 2016 survey (kaggle.com/osmi/mental-health-in-tech-2016).

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

df16[['Age', 'Gender', 'treatment', 'family_history', 'benefits', 'anonymity', 'remote_work', 'no_employees']].head()

# ### Chart 16 - Treatment Rate: 2014 vs 2016
# **Insight:** Rose from ~50.6% to ~58.5% - directional trend, not the same respondents tracked over time.

rate_2014 = (df['treatment'] == 'Yes').mean() * 100
rate_2016 = (df16['treatment'] == 'Yes').mean() * 100

plt.figure(figsize=(6, 5))
bars = plt.bar(['2014', '2016'], [rate_2014, rate_2016], color=['#4C72B0', '#DD8452'])
plt.title('% of Respondents Who Sought Treatment: 2014 vs 2016')
plt.ylabel('% Sought Treatment')
plt.ylim(0, 100)
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f'{b.get_height():.1f}%', ha='center')
plt.savefig('eda_outputs/16_treatment_rate_2014_vs_2016.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 17 - Benefits & Anonymity Awareness: 2014 vs 2016
# **Insight:** "Don't know" share stayed large for both years (largest for anonymity, ~65%) - the awareness gap persisted rather than improving.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in zip(axes, ['benefits', 'anonymity'],
                            ['Employer Provides Benefits', 'Anonymity Protected']):
    order = ['Yes', "Don't know", 'No']
    d14 = df[col].value_counts(normalize=True).reindex(order) * 100
    d16 = df16[col].value_counts(normalize=True).reindex(order) * 100
    x = np.arange(len(order))
    width = 0.35
    ax.bar(x - width/2, d14.values, width, label='2014', color='#4C72B0')
    ax.bar(x + width/2, d16.values, width, label='2016', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_ylabel('% of Respondents')
    ax.legend()
plt.tight_layout()
plt.savefig('eda_outputs/17_benefits_anonymity_2014_vs_2016.png', dpi=150, bbox_inches='tight')
plt.show()

# ### Chart 18 - Company Size: 2014 vs 2016
# **Insight:** Broadly similar mix both years - trends above aren't just a sampling artifact.

size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
d14 = df['no_employees'].value_counts(normalize=True).reindex(size_order) * 100
d16 = df16['no_employees'].value_counts(normalize=True).reindex(size_order) * 100

x = np.arange(len(size_order))
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, d14.values, width, label='2014', color='#4C72B0')
plt.bar(x + width/2, d16.values, width, label='2016', color='#DD8452')
plt.xticks(x, size_order, rotation=20)
plt.ylabel('% of Respondents')
plt.title('Company Size Distribution: 2014 vs 2016')
plt.legend()
plt.tight_layout()
plt.savefig('eda_outputs/18_company_size_2014_vs_2016.png', dpi=150, bbox_inches='tight')
plt.show()