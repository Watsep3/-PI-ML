import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import ttest_ind

sns.set_theme(style="whitegrid")  # Pour un style plus propre dans les graphiques

def main():
    st.title("Analyse de données : Bien-être mental, habitudes de vie, et plus")

    # -------------------------------------------------------------------------
    # CHARGEMENT AUTOMATIQUE DU FICHIER CSV (AUCUNE SAISIE D'UTILISATEUR)
    # -------------------------------------------------------------------------
    try:
        data = pd.read_csv("data/CleanedData.csv")
        st.success("Données chargées automatiquement depuis data/CleanedData.csv !")
        st.write("**Aperçu des données :**")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()  # Stop l'exécution si on ne peut pas charger le CSV

    # =========================================================================
    # PARTIE 2 : VISUALISATIONS EXPLORATOIRES
    # =========================================================================
    with st.expander("2. Visualisations Exploratoires", expanded=False):
        # 2.1 Distribution de l'âge
        st.markdown("### Distribution de l'âge")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.histplot(data["Age"], bins=8, kde=True, color="skyblue", ax=ax1)
        ax1.set_title("Age Distribution")
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        # 2.2 Distribution des notes de santé mentale (Mental_Health_Rating)
        st.markdown("### Distribution des notes de santé mentale (Mental_Health_Rating)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(data['Mental_Health_Rating'], bins=10, kde=True, ax=ax2)
        ax2.set_title('Distribution of Mental Health Ratings')
        ax2.set_xlabel('Rating (1-10)')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

        # 2.3 Boxplot : Mental Health Rating par genre
        st.markdown("### Mental Health Rating par genre (Boxplot)")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Gender", y="Mental_Health_Rating", data=data, palette="pastel", ax=ax3)
        ax3.set_title("Boxplot of Mental Health Ratings by Gender")
        ax3.set_xlabel("Gender")
        ax3.set_ylabel("Mental Health Rating")
        st.pyplot(fig3)

        # 2.4 Densité : Screen Time
        st.markdown("### Densité : Screen Time (Screen_Time_Hours)")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data["Screen_Time_Hours"], shade=True, color="purple", ax=ax4)
        ax4.set_title("Density Plot of Screen Time")
        ax4.set_xlabel("Hours of Screen Time")
        ax4.set_ylabel("Density")
        st.pyplot(fig4)

        # 2.5 Pairplot global (variables numériques)
        st.markdown("### Pairplot (toutes les variables numériques)")
        fig5 = sns.pairplot(data)
        st.pyplot(fig5)

        # 2.6 Matrice de corrélation (Heatmap)
        st.markdown("### Matrice de corrélation (Heatmap)")
        correlation_matrix = data.select_dtypes(include="number").corr()
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax6)
        ax6.set_title("Heatmap of Correlation Matrix")
        st.pyplot(fig6)

    # =========================================================================
    # PARTIE 3 : VISUALISATIONS DE RELATIONS CLÉS
    # =========================================================================
    with st.expander("3. Visualisations de relations clés", expanded=False):
        # 3.1 Work-Study Balance vs Mental Health
        st.markdown("### Work-Study Balance vs Mental Health Rating")
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=data['Work_Study_Balance'], y=data['Mental_Health_Rating'], ax=ax7)
        ax7.set_title('Work-Study Balance vs Mental Health Rating')
        ax7.set_xlabel('Work-Study Balance (1-10)')
        ax7.set_ylabel('Mental Health Rating (1-10)')
        st.pyplot(fig7)

        # 3.2 Social Activity Frequency vs Mental Health
        st.markdown("### Social Activity Frequency vs Mental Health (Hexbin)")
        fig8, ax8 = plt.subplots(figsize=(8, 6))
        hb = ax8.hexbin(
            x=data['Social_Activity_Frequency'],
            y=data['Mental_Health_Rating'],
            gridsize=30,
            cmap="Reds",
            mincnt=1
        )
        fig8.colorbar(hb, label='Frequency')
        ax8.set_title('Social Activity Frequency vs Mental Health Rating')
        ax8.set_xlabel('Social Activity Frequency (1-10)')
        ax8.set_ylabel('Mental Health Rating (1-10)')
        st.pyplot(fig8)

        # 3.3 Screen Time vs Mental Health
        st.markdown("### Screen Time vs Mental Health Rating")
        fig9, ax9 = plt.subplots(figsize=(12, 8))
        sns.boxplot(x=data['Screen_Time_Hours'], y=data['Mental_Health_Rating'], ax=ax9)
        ax9.set_title('Screen Time Hours vs Mental Health Rating')
        ax9.set_xlabel('Screen Time Hours')
        ax9.set_ylabel('Mental Health Rating')
        plt.xticks(rotation=45)
        st.pyplot(fig9)

        # 3.4 Social Media Impact vs Mental Health
        st.markdown("### Social Media Impact vs Mental Health (Hexbin)")
        fig10, ax10 = plt.subplots(figsize=(8, 6))
        hb2 = ax10.hexbin(
            x=data['Social_Media_Impact'],
            y=data['Mental_Health_Rating'],
            gridsize=30,
            cmap="Reds",
            mincnt=1
        )
        fig10.colorbar(hb2, label='Frequency')
        ax10.set_title('Social Media Impact vs Mental Health Rating')
        ax10.set_xlabel('Social Media Impact (1-10)')
        ax10.set_ylabel('Mental Health Rating (1-10)')
        st.pyplot(fig10)

        # 3.5 Sleep Hours vs Mental Health
        st.markdown("### Sleep Hours vs Mental Health Rating")
        fig11, ax11 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=data['Sleep_Hours'], y=data['Mental_Health_Rating'], ax=ax11)
        ax11.set_title('Sleep Hours vs Mental Health Rating')
        ax11.set_xlabel('Sleep Hours')
        ax11.set_ylabel('Mental Health Rating')
        st.pyplot(fig11)

        # 3.6 Social Media Impact vs Work-Study Balance
        st.markdown("### Social Media Impact vs Work-Study Balance (Hexbin)")
        fig12, ax12 = plt.subplots(figsize=(8, 6))
        hb3 = ax12.hexbin(
            x=data['Social_Media_Impact'],
            y=data['Work_Study_Balance'],
            gridsize=30,
            cmap="Reds",
            mincnt=1
        )
        fig12.colorbar(hb3, label='Frequency')
        ax12.set_title('Social Media Impact vs Work-Study Balance')
        ax12.set_xlabel('Social Media Impact (1-10)')
        ax12.set_ylabel('Work-Study Balance (1-10)')
        st.pyplot(fig12)

    # =========================================================================
    # PARTIE 4 : PROBABILITÉS ET STATISTIQUES DESCRIPTIVES
    # =========================================================================
    with st.expander("4. Probabilités et statistiques descriptives", expanded=False):
        st.write("### Statistiques Descriptives")
        st.write(data.describe())

        st.markdown("### Substance Use Probabilities by Gender")
        substance_use_by_gender = data.groupby(['Gender', 'Substance_Use']).size().unstack(fill_value=0)
        total_by_gender = substance_use_by_gender.sum(axis=1)
        probabilities = substance_use_by_gender.div(total_by_gender, axis=0)

        st.dataframe(probabilities)

        fig13, ax13 = plt.subplots(figsize=(6, 4))
        probabilities.plot(kind='bar', stacked=True, ax=ax13)
        ax13.set_title('Substance Use Probabilities by Gender')
        ax13.set_xlabel('Gender')
        ax13.set_ylabel('Probability')
        ax13.legend(title='Substance Use', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig13)

        total_by_gender_val = data['Gender'].value_counts()
        non_nan_substance_use = data[data['Substance_Use'].notna()].groupby('Gender')['Substance_Use'].count()
        probability_non_nan = non_nan_substance_use / total_by_gender_val

        st.markdown("**Probability of non-NaN substance use by gender :**")
        st.write(probability_non_nan)

        st.markdown("**Percentage of non-NaN substance use by gender :**")
        st.write(probability_non_nan * 100)

    # =========================================================================
    # PARTIE 5 : HYPOTHÈSES ET TESTS STATISTIQUES
    # =========================================================================
    with st.expander("5. Hypothèses et Tests Statistiques", expanded=False):
        # --------------------------------------------------------------------
        # HYPOTHÈSE 1: Screen Time vs Mental Health
        # --------------------------------------------------------------------
        st.markdown("### Hypothèse 1 : Screen Time vs Mental Health")
        st.markdown("""
        **Null Hypothesis (H0):**  
        Il n'y a pas de relation entre le temps d'écran (Screen_Time_Hours) et la santé mentale (Mental_Health_Rating).

        **Alternative Hypothesis (H1):**  
        Plus le temps d'écran augmente, plus la note de santé mentale diminue.
        """)

        X = data[['Screen_Time_Hours']]
        y = data['Mental_Health_Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        coef = model.coef_[0]
        intercept = model.intercept_
        st.markdown(f"- **Coefficient (pente)** : `{coef:.4f}`")
        st.markdown(f"- **Intercept** : `{intercept:.4f}`")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.markdown(f"- **Mean Squared Error (MSE)** : `{mse:.4f}`")
        st.markdown(f"- **R-squared (R²)** : `{r2:.4f}`")

        X_with_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_with_const).fit()
        st.markdown("**Régression (Statsmodels OLS)**")
        st.text(ols_model.summary())

        fig14, ax14 = plt.subplots(figsize=(8, 6))
        ax14.scatter(X, y, color='blue', label='Data Points')
        ax14.plot(X, model.predict(X), color='red', label='Regression Line')
        ax14.set_title('Screen Time vs Mental Health Rating')
        ax14.set_xlabel('Screen Time (Hours)')
        ax14.set_ylabel('Mental Health Rating')
        ax14.legend()
        st.pyplot(fig14)

        st.markdown("""
        **Interprétation :**  
        - La relation est significative (p < 0.05), mais R² = 0.064 (faible pouvoir explicatif).  
        - La pente négative (env. -0.126) suggère qu’une hausse du Screen Time est associée à une légère baisse de la santé mentale.
        """)

        # --------------------------------------------------------------------
        # HYPOTHÈSE 2: Work-Study Balance vs Mental Health Rating
        # --------------------------------------------------------------------
        st.markdown("### Hypothèse 2 : Work-Study Balance vs Mental Health Rating")
        st.markdown("""
        **Null Hypothesis (H0):**  
        Il n'y a pas de corrélation entre Work-Study Balance et la santé mentale.

        **Alternative Hypothesis (H1):**  
        Une meilleure Work-Study Balance est associée à une meilleure note de santé mentale.
        """)

        stat_balance, p_balance = stats.shapiro(data['Work_Study_Balance'])
        stat_mental, p_mental = stats.shapiro(data['Mental_Health_Rating'])

        st.markdown(f"- **P-value (Work-Study Balance)** : `{p_balance:.4g}`")
        st.markdown(f"- **P-value (Mental_Health_Rating)** : `{p_mental:.4g}`")

        if p_balance > 0.05 and p_mental > 0.05:
            st.markdown("> Les variables sont normales : **corrélation de Pearson**.")
        else:
            st.markdown("> Les variables ne sont pas normales : **corrélation de Spearman**.")

        corr_val, p_value_corr = stats.spearmanr(data['Work_Study_Balance'], data['Mental_Health_Rating'])
        st.markdown(f"- **Spearman Correlation** : `{corr_val:.4f}`")
        st.markdown(f"- **p-value** : `{p_value_corr:.4g}`")

        if p_value_corr < 0.05:
            st.markdown("=> Corrélation **positive et significative**.")
        else:
            st.markdown("=> **Aucune** corrélation significative.")

        fig15, ax15 = plt.subplots(figsize=(8, 6))
        hb4 = ax15.hexbin(
            x=data['Work_Study_Balance'],
            y=data['Mental_Health_Rating'],
            gridsize=30,
            cmap="Reds",
            mincnt=1
        )
        fig15.colorbar(hb4, label='Frequency')
        ax15.set_title('Work-Study Balance vs Mental Health Rating')
        ax15.set_xlabel('Work-Study Balance (1-10)')
        ax15.set_ylabel('Mental Health Rating (1-10)')
        st.pyplot(fig15)

        st.markdown("""
        **Interprétation :**  
        - Corrélation modérée (~0.35) et p-value très faible => Relation significative.  
        - Une meilleure Work-Study Balance est associée à une meilleure santé mentale.
        """)

        # --------------------------------------------------------------------
        # HYPOTHÈSE 3: Social Media Impact vs Mental Health Rating
        # --------------------------------------------------------------------
        st.markdown("### Hypothèse 3 : Social Media Impact vs Mental Health Rating")
        st.markdown("""
        **Null Hypothesis (H0):**  
        Pas de relation entre l'impact des réseaux sociaux et la santé mentale.

        **Alternative Hypothesis (H1):**  
        Plus l'impact des réseaux sociaux est élevé, plus la santé mentale est affectée négativement.
        """)

        stat_media, p_media = stats.shapiro(data['Social_Media_Impact'])
        st.markdown(f"- **P-value (Social_Media_Impact)** : `{p_media:.4g}`")
        st.markdown(f"- **P-value (Mental_Health_Rating)** : `{p_mental:.4g}`")

        if p_media > 0.05 and p_mental > 0.05:
            st.markdown("> **Corrélation de Pearson**.")
        else:
            st.markdown("> **Corrélation de Spearman**.")

        corr_sm, p_value_sm = stats.spearmanr(data['Social_Media_Impact'], data['Mental_Health_Rating'])
        st.markdown(f"- **Spearman Correlation** : `{corr_sm:.4f}`")
        st.markdown(f"- **p-value** : `{p_value_sm:.4g}`")

        if p_value_sm < 0.05:
            st.markdown("=> Corrélation **négative et significative**.")
        else:
            st.markdown("=> **Aucune** corrélation significative.")

        st.markdown("""
        **Interprétation :**  
        - Relation statistiquement significative mais de faible amplitude.  
        - L’impact des réseaux sociaux seul n’explique pas entièrement la santé mentale.
        """)

        # --------------------------------------------------------------------
        # HYPOTHÈSE 4: Financial Stability vs Work-Study Balance
        # --------------------------------------------------------------------
        st.markdown("### Hypothèse 4 : Financial Stability vs Work-Study Balance")
        st.markdown("""
        **Null Hypothesis (H0):**  
        Aucune relation entre la stabilité financière et Work-Study Balance.

        **Alternative Hypothesis (H1):**  
        Plus la stabilité financière est élevée, meilleure est la Work-Study Balance.
        """)

        stat_financial, p_financial = stats.shapiro(data['Financial_Situation'])
        st.markdown(f"- **P-value (Financial_Situation)** : `{p_financial:.4g}`")
        st.markdown(f"- **P-value (Work_Study_Balance)** : `{p_balance:.4g}`")

        if p_financial > 0.05 and p_balance > 0.05:
            st.markdown("> **Corrélation de Pearson**.")
        else:
            st.markdown("> **Corrélation de Spearman**.")

        corr_fin, p_fin = stats.spearmanr(data['Financial_Situation'], data['Work_Study_Balance'])
        st.markdown(f"- **Spearman Correlation** : `{corr_fin:.4f}`")
        st.markdown(f"- **p-value** : `{p_fin:.4g}`")

        if p_fin < 0.05:
            st.markdown("=> Corrélation **positive et significative**.")
        else:
            st.markdown("=> **Aucune** corrélation significative.")

        st.markdown("""
        **Interprétation :**  
        - L’aspect financier influence positivement la Work-Study Balance, mais faiblement.  
        - D'autres facteurs peuvent être plus déterminants.
        """)

        # --------------------------------------------------------------------
        # HYPOTHÈSE 5: Consultation Professionnelle vs Mental Health
        # --------------------------------------------------------------------
        st.markdown("### Hypothèse 5 : Consultation Professionnelle vs Mental Health Rating")
        st.markdown("""
        **Null Hypothesis (H0):**  
        Pas de différence de santé mentale entre ceux qui consultent et ceux qui ne consultent pas.

        **Alternative Hypothesis (H1):**  
        Ceux qui consultent un professionnel ont une meilleure santé mentale.
        """)

        group_not_consulted = data[data['Consulted_Professional'] == 'No']['Mental_Health_Rating']
        group_consulted = data[
            (data['Consulted_Professional'] == 'Yes') |
            (data['Consulted_Professional'] == 'Prefer not to say')
        ]['Mental_Health_Rating']

        stat_var, p_var = stats.levene(group_consulted, group_not_consulted)
        st.markdown(f"- **Levene Test (égalité des variances), p-value** : `{p_var:.4g}`")

        equal_var_test = (p_var > 0.05)
        if equal_var_test:
            st.markdown("> Variances supposées égales (T-test standard).")
        else:
            st.markdown("> Variances inégales (T-test de Welch).")

        t_stat, p_value_t = stats.ttest_ind(group_consulted, group_not_consulted, equal_var=equal_var_test)
        st.markdown(f"- **T-Statistic** : `{t_stat:.4f}`")
        st.markdown(f"- **p-value** : `{p_value_t:.4g}`")

        if p_value_t < 0.05:
            st.markdown("=> Différence **significative** entre les deux groupes.")
        else:
            st.markdown("=> **Pas** de différence significative.")

        fig16, ax16 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=data['Consulted_Professional'], y=data['Mental_Health_Rating'], ax=ax16)
        ax16.set_title('Mental Health Ratings by Consultation with a Mental Health Professional')
        ax16.set_xlabel('Consulted a Mental Health Professional')
        ax16.set_ylabel('Mental Health Rating (1-10)')
        st.pyplot(fig16)

        st.markdown("""
        **Interprétation :**  
        - Aucune preuve solide (p > 0.05) que la consultation améliore la santé mentale dans ce jeu de données.  
        - Les boxplots indiquent toutefois une tendance légèrement positive.
        """)

        # --------------------------------------------------------------------
        # HYPOTHÈSE 6: Sleep Hours vs Mental Health
        # --------------------------------------------------------------------
        st.markdown("### Hypothèse 6 : Sleep Hours vs Mental Health Rating")
        st.markdown("""
        **Null Hypothesis (H0):**  
        Aucune relation entre les heures de sommeil et la santé mentale.

        **Alternative Hypothesis (H1):**  
        Plus on dort, meilleure est la santé mentale.
        """)

        corr_sleep, p_val_sleep = stats.pearsonr(data['Sleep_Hours'], data['Mental_Health_Rating'])
        st.markdown(f"- **Correlation coefficient** : `{corr_sleep:.4f}`")
        st.markdown(f"- **p-value** : `{p_val_sleep:.4g}`")

        if p_val_sleep < 0.05:
            st.markdown("=> **Corrélation significative**.")
        else:
            st.markdown("=> **Aucune** corrélation significative.")

        st.markdown("""
        **Interprétation :**  
        - Corrélation très faible et p-value > 0.05 => Pas de preuve de relation significative dans ces données.
        """)

    # =========================================================================
    # PARTIE 6 : INTERVALLES DE CONFIANCE
    # =========================================================================
    with st.expander("6. Intervalles de Confiance", expanded=False):
        st.markdown("### 6.1 CI pour la moyenne de Mental Health Rating")
        mental_health_ratings = data['Mental_Health_Rating']
        mean_mhr = np.mean(mental_health_ratings)
        std_err_mhr = stats.sem(mental_health_ratings)
        ci_mental_health = stats.t.interval(
            confidence=0.95,
            df=len(mental_health_ratings) - 1,
            loc=mean_mhr,
            scale=std_err_mhr
        )
        st.write("IC 95% :", ci_mental_health)

        st.markdown("### 6.2 CI pour la proportion consultant un professionnel")
        consulted_yes_count = (data['Consulted_Professional'] == 'Yes').sum()
        total_responses = len(data['Consulted_Professional'])
        proportion_consulted = consulted_yes_count / total_responses
        se_consulted = np.sqrt((proportion_consulted * (1 - proportion_consulted)) / total_responses)
        ci_consulted = (
            proportion_consulted - stats.norm.ppf(0.975) * se_consulted,
            proportion_consulted + stats.norm.ppf(0.975) * se_consulted
        )
        st.write("IC 95% :", ci_consulted)

        st.markdown("### 6.3 CI pour la moyenne d'heures de sommeil")
        sleep_hours = data['Sleep_Hours']
        mean_sleep = np.mean(sleep_hours)
        std_err_sleep = stats.sem(sleep_hours)
        ci_sleep_hours = stats.t.interval(
            confidence=0.95,
            df=len(sleep_hours) - 1,
            loc=mean_sleep,
            scale=std_err_sleep
        )
        st.write("IC 95% :", ci_sleep_hours)

        st.markdown("### 6.4 CI pour la proportion d'hommes ne consommant qu'alcool")
        men_only_drink = data[(data['Gender'] == 'Male') & (data['Substance_Use'] == 'Alcohol')]
        total_men = len(data[data['Gender'] == 'Male'])
        proportion_men_only_drink = len(men_only_drink) / total_men if total_men > 0 else 0
        if total_men > 0:
            se_men_only_drink = np.sqrt((proportion_men_only_drink * (1 - proportion_men_only_drink)) / total_men)
        else:
            se_men_only_drink = 0
        ci_men_only_drink = (
            proportion_men_only_drink - stats.norm.ppf(0.975) * se_men_only_drink,
            proportion_men_only_drink + stats.norm.ppf(0.975) * se_men_only_drink
        )
        st.write("IC 95% :", ci_men_only_drink)

        st.markdown("### 6.5 CI pour la proportion de femmes consommant du tabac (90%)")
        women_tobacco = data[(data['Gender'] == 'Female') & (data['Substance_Use'] == 'Tobacco')]
        total_women = len(data[data['Gender'] == 'Female'])
        proportion_women_tobacco = len(women_tobacco) / total_women if total_women > 0 else 0
        if total_women > 0:
            se_women_tobacco = np.sqrt((proportion_women_tobacco * (1 - proportion_women_tobacco)) / total_women)
        else:
            se_women_tobacco = 0
        ci_women_tobacco = (
            proportion_women_tobacco - stats.norm.ppf(0.95) * se_women_tobacco,
            proportion_women_tobacco + stats.norm.ppf(0.95) * se_women_tobacco
        )
        st.write("IC 90% :", ci_women_tobacco)

    # =========================================================================
    # PARTIE 7 : TESTS D'HYPOTHÈSES SUPPLÉMENTAIRES
    # =========================================================================
    with st.expander("7. Tests d'hypothèses supplémentaires", expanded=False):
        st.markdown("### Test 1 : Différence de Mental Health Ratings selon le Genre")
        male_mhr = data[data['Gender'] == 'Male']['Mental_Health_Rating']
        female_mhr = data[data['Gender'] == 'Female']['Mental_Health_Rating']
        t_stat_gender, p_value_gender = stats.ttest_ind(male_mhr, female_mhr, equal_var=False)
        st.write(f"t-statistic = {t_stat_gender:.4f}, p-value = {p_value_gender:.4g}")
        if p_value_gender < 0.05:
            st.write("=> Différence **significative** entre hommes et femmes.")
        else:
            st.write("=> Aucune différence significative.")

        st.markdown("### Test 2 : Effet de consulter un professionnel sur Mental Health Ratings")
        consulted_yes_2 = data[data['Consulted_Professional'] == 'Yes']['Mental_Health_Rating']
        consulted_no_2 = data[data['Consulted_Professional'] == 'No']['Mental_Health_Rating']
        t_stat_consult, p_value_consult = stats.ttest_ind(consulted_yes_2, consulted_no_2, equal_var=False)
        st.write(f"t-statistic = {t_stat_consult:.4f}, p-value = {p_value_consult:.4g}")
        if p_value_consult < 0.05:
            st.write("=> Effet significatif de la consultation sur la santé mentale.")
        else:
            st.write("=> Pas d'effet significatif.")

        st.markdown("### Test 3 : Corrélation entre Social Media Impact et Mental Health Ratings")
        social_media_impact = data['Social_Media_Impact']
        rating = data['Mental_Health_Rating']
        corr_smi, p_smi = stats.pearsonr(social_media_impact, rating)
        st.write(f"Corrélation = {corr_smi:.4f}, p-value = {p_smi:.4g}")
        if p_smi < 0.05:
            st.write("=> Corrélation significative.")
        else:
            st.write("=> Aucune corrélation significative.")

        st.markdown("### Test 4 : Impact de la fréquence d'exercice sur le temps de sommeil (ANOVA)")
        exercise_groups = data.groupby('Exercise_Frequency')['Sleep_Hours'].apply(list)
        f_stat_ex, p_value_ex = stats.f_oneway(*exercise_groups)
        st.write(f"F-statistic = {f_stat_ex:.4f}, p-value = {p_value_ex:.4g}")
        if p_value_ex < 0.05:
            st.write("=> La fréquence d'exercice impacte significativement les heures de sommeil.")
        else:
            st.write("=> Pas d'impact significatif.")

        st.markdown("### Test 5 : Substance Use Frequency vs. Social Activity Frequency (ANOVA)")
        substance_groups = data.groupby('Substance_Use')['Social_Activity_Frequency'].apply(list)
        f_stat_sub, p_value_sub = stats.f_oneway(*substance_groups)
        st.write(f"F-statistic = {f_stat_sub:.4f}, p-value = {p_value_sub:.4g}")
        if p_value_sub < 0.05:
            st.write("=> La substance consommée impacte significativement la fréquence d'activité sociale.")
        else:
            st.write("=> Pas d'impact significatif.")
            
        # -----------------------------
        # NOUVEAU TEST : Balanced Diet
        # -----------------------------
        st.markdown("### Test 6 : Balanced Diet vs Other Diets (T-test)")

        # On retire les éventuelles valeurs NaN dans Diet et Mental_Health_Rating
        data2 = data.dropna(subset=['Diet', 'Mental_Health_Rating'])

        # Groupes : Balanced vs non-Balanced
        balanced_diet_group = data2[data2['Diet'] == 'Balanced']['Mental_Health_Rating']
        other_diet_group = data2[data2['Diet'] != 'Balanced']['Mental_Health_Rating']

        # Test t_indépendant (Welch avec equal_var=False)
        t_stat, p_value = ttest_ind(balanced_diet_group, other_diet_group, equal_var=False)

        # Affichage des résultats dans Streamlit
        st.write("**T-Test: Balanced Diet vs Other Diets**")
        st.write(f"T-Statistic: `{t_stat:.4f}`")
        st.write(f"P-Value: `{p_value:.4f}`")

        if p_value < 0.05:
            st.write("**Conclusion :** Le p-value est significatif (p < 0.05). \
            Il existe une différence significative de mental health ratings entre les régimes équilibrés et les autres régimes.")
        else:
            st.write("**Conclusion :** Le p-value n'est pas significatif (p ≥ 0.05). \
            Aucune preuve solide d'une différence dans les mental health ratings entre les régimes équilibrés et les autres régimes.")
    
        

# ----------------------------------------------------------------------------
# Exécuter l'application
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
