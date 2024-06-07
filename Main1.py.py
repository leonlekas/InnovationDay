import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



#Paths
path_train=r'D:\Python\Innovationday\InnovationDay\fifa_players.csv'
df_train=pd.read_csv(path_train)

#df_test=pd.read_csv(path_test)
columns_to_delete = ['name','full_name','birth_date','overall_rating','potential','national_rating','national_team','nationality','national_jersey_number','national_team_position'] 
df_train = df_train.drop(columns=columns_to_delete)

def data_statistics(df):
    # Ausgabe der ersten paar Zeilen des DataFrames
    print("Head of the DataFrame:")
    print(df.head())
    print("\n")

    # Allgemeine Statistiken für die 'SalePrice'-Spalte
    print("General statistics for 'value_euro':")
    print(df['value_euro'].describe())
    print("\n")

    # Histogramm für die Verteilung der Verkaufspreise
    plt.figure(figsize=(8, 6))
    plt.hist(df['value_euro'], color='skyblue', edgecolor='black')
    plt.title('Distribution of value_euro')
    plt.xlabel('value_euro')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Korrelationsmatrix
    korrelation_matrix = df.corr(numeric_only=True)
    print("Correlation Matrix:")
    print(korrelation_matrix)
    print("\n")

    # Visualisierung der Korrelationsmatrix als Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(korrelation_matrix, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

data_statistics(df_train)


def data_correlation(df):
    correlation_matrix = df.corr()
    correlation_with_value_euro = correlation_matrix['value_euro'].sort_values(ascending=False)

    # Die zehn Merkmale mit den höchsten positiven Korrelationen
    top_positive_correlations = correlation_with_value_euro[1:11]

    # Die zehn Merkmale mit den höchsten negativen Korrelationen
    top_negative_correlations = correlation_with_value_euro.tail(10)
    extended_df = pd.concat([top_positive_correlations, top_negative_correlations], axis=0)

    return correlation_with_value_euro,  extended_df

# Positionen kategorisieren
def categorize_positions(positions):
    defense_positions = {'CB', 'LB', 'RB', 'LWB', 'RWB', 'SW'}
    goalkeeper_positions = {'GK'}
    attack_positions = {'ST', 'CF', 'LW', 'RW'}
    midfield_positions = {'CM', 'CDM', 'CAM', 'LM', 'RM'}

    positions = positions.split(',')
    for pos in positions:
        if pos in defense_positions:
            return 'defense'
        elif pos in goalkeeper_positions:
            return 'goalkeeper'
        elif pos in attack_positions:
            return 'attack'
        elif pos in midfield_positions:
            return 'midfield'
    return 'unknown'

# Kategorie Spalte hinzufügen
df_train['category'] = df_train['positions'].apply(categorize_positions)

# Subdatensätze erstellen
defense_df = df_train[df_train['category'] == 'defense']
goalkeeper_df = df_train[df_train['category'] == 'goalkeeper']
attack_df = df_train[df_train['category'] == 'attack']
midfield_df = df_train[df_train['category'] == 'midfield']


defense_df=defense_df.drop(columns=['positions'])
goalkeeper_df=goalkeeper_df.drop(columns=['positions'])
attack_df=attack_df.drop(columns=['positions'])
midfield_df=midfield_df.drop(columns=['positions'])

data_statistics(defense_df)
data_statistics(goalkeeper_df)
data_statistics(attack_df)
data_statistics(midfield_df)




