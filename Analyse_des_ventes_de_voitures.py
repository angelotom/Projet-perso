import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

os.environ["DISPLAY"] = ""

def load_data(path):
    try:
        df = pd.read_csv(path)
        print("Données chargées avec succès.")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None


def clean_data(df):
    # Suppression des doublons
    df = df.drop_duplicates()

    # Normalisation des chaînes
    string_cols = ['make', 'model', 'transmission', 'body']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # Conversion des types
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['sellingprice'] = pd.to_numeric(df['sellingprice'], errors='coerce')

    # Suppression des lignes incohérentes
    df = df[(df['sellingprice'] > 0) & (df['year'] > 1980) & (df['year'] < 2025)]

    # Remplissage des valeurs manquantes
    if 'make' in df.columns and 'model' in df.columns:
        df['make'] = df.groupby('model')['make'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    df['make'].fillna('unknown', inplace=True)

    return df


def analyse_data(df):
    print("\nAperçu des données:")
    print(df.head())
    print("\nStatistiques numériques:")
    print(df.describe())
    print("\nInformations générales:")
    print(df.info())
    print("\nValeurs manquantes (%):")
    print((df.isnull().sum() / len(df)) * 100)


def visualisations(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Marques les plus vendues
    top_marques = df['make'].value_counts().head(5)
    plt.figure(figsize=(10, 6))
    plt.bar(top_marques.index, top_marques.values, color='blue')
    plt.title("Top 5 des marques les plus vendues")
    plt.xlabel("Marque")
    plt.ylabel("Nombre de ventes")
    plt.savefig(f"{output_dir}/top_marques.png")
    plt.close()

    # Évolution des ventes par année
    ventes_annee = df['year'].value_counts().sort_index()
    ventes_annee.plot(kind='line', marker='o', figsize=(10, 5))
    plt.title("Évolution des ventes par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre de ventes")
    plt.grid()
    plt.savefig(f"{output_dir}/ventes_par_annee.png")
    plt.close()

    # Prix moyen par année
    prix_moyen = df.groupby('year')['sellingprice'].mean()
    prix_moyen.plot(kind='line', marker='o', figsize=(10, 5))
    plt.title("Prix moyen des véhicules par année")
    plt.xlabel("Année")
    plt.ylabel("Prix moyen")
    plt.grid()
    plt.savefig(f"{output_dir}/prix_moyen_par_annee.png")
    plt.close()

    # Distribution des prix
    df['sellingprice'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black', figsize=(10, 6))
    plt.title("Distribution des prix de vente")
    plt.xlabel("Prix")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.savefig(f"{output_dir}/distribution_prix.png")
    plt.close()

    # Ventes par type de carrosserie
    df.groupby(['year', 'body']).size().unstack().plot(kind='line', figsize=(12, 6))
    plt.title("Évolution des ventes par type de carrosserie")
    plt.xlabel("Année")
    plt.ylabel("Nombre de ventes")
    plt.legend(title="Type de carrosserie")
    plt.grid()
    plt.savefig(f"{output_dir}/ventes_par_carrosserie.png")
    plt.close()

    # Heatmap corrélation
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()


def generate_report(df, output_path="outputs/rapport_vehicules.html"):
    try:
        rapport = ProfileReport(df, title="Rapport d'Analyse des Ventes de Véhicules", explorative=True)
        rapport.to_file(output_path)
        print(f"\nRapport HTML généré : {output_path}")
    except Exception as e:
        print(f"Erreur lors de la génération du rapport : {e}")


def resume(df):
    print("\n--- Résumé de l'analyse ---")
    print(f"Nombre total d'enregistrements : {len(df)}")
    print(f"Années couvertes : {df['year'].min()} à {df['year'].max()}")
    print(f"Nombre de marques uniques : {df['make'].nunique()}")
    print(f"Prix moyen : {df['sellingprice'].mean():,.2f}")


def main():
    data_path = r"C:\Users\THOMAS\Downloads\usa_vehicle_sales_data\car_prices.csv"
    df = load_data(data_path)

    if df is not None:
        df = clean_data(df)
        analyse_data(df)
        visualisations(df)
        generate_report(df)
        resume(df)
        print("\nScript terminé avec succès.")


if __name__ == "__main__":
    main()
