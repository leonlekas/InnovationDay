import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#Paths
path_train=r'C:\Users\Jonas\Documents\Python\New Project\Kaggle\House_Prices\Data\train.csv'
path_test=r'C:\Users\Jonas\Documents\Python\New Project\Kaggle\House_Prices\Data\test.csv'
df_train=pd.read_csv(path_train)
df_test=pd.read_csv(path_test)

def data_statistics(df):
    # Ausgabe der ersten paar Zeilen des DataFrames
    print("Head of the DataFrame:")
    print(df.head())
    print("\n")

    # Allgemeine Statistiken für die 'SalePrice'-Spalte
    print("General statistics for 'SalePrice':")
    print(df['SalePrice'].describe())
    print("\n")

    # Histogramm für die Verteilung der Verkaufspreise
    plt.figure(figsize=(8, 6))
    plt.hist(df['SalePrice'], color='skyblue', edgecolor='black')
    plt.title('Distribution of Sale Prices')
    plt.xlabel('Sale Price')
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


def data_preparation(df_train, df_test):
    # Umwandlung in numerische Werte mit Mapping

    # Definition der Mapping-Dictionaries
    mappings = {
        'CentralAir': {'N': 0, 'Y': 1},
        'Street': {'Grvl': 1, 'Pave': 0},
        'Alley': {'Grvl': 1, 'Pave': 0, 'NA': 2},
        'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
        'LandContour': {'Low': 1, 'Lvl': 2, 'Bnk': 3, 'HLS': 4},
        'Utilities': {'AllPub': 1, 'NoSewr': 2, 'NoSeWa': 3, 'ELO': 4},
        'LandSlope': {'Gtl': 1, 'Mod': 2, 'Sev': 3},
        'BldgType': {'1Fam': 1, '2FmCon': 2, 'Duplx': 3, 'TwnhsE': 4, 'TwnhsI': 5},
        'HouseStyle': {'1Story': 1, '1.5Fin': 2, '1.5Unf': 3, '2Story': 4, '2.5Fin': 5, '2.5Unf': 6, 'SFoyer': 7, 'SLvl': 8},
        'ExterQual': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5},
        'ExterCond': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5},
        'HeatingQC': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5},
        'KitchenQual': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5},
        'BsmtQual': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5, 'NA': 6},
        'BsmtCond': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5, 'NA': 6},
        'GarageCond': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5, 'NA': 6},
        'GarageQual': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5, 'NA': 6},
        'FireplaceQu': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5, 'NA': 6},
        'BsmtExposure': {'Gd': 1, 'Av': 2, 'Mn': 3, 'No': 4, 'NA': 5},
        'BsmtFinType1': {'GLQ': 1, 'ALQ': 2, 'BLQ': 3, 'Rec': 4, 'LwQ': 5, 'Unf': 6, 'NA': 7},
        'BsmtFinType2': {'GLQ': 1, 'ALQ': 2, 'BLQ': 3, 'Rec': 4, 'LwQ': 5, 'Unf': 6, 'NA': 7},
        'Electrical': {'SBrkr': 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5},
        'Functional': {'Typ': 1, 'Min1': 2, 'Min2': 3, 'Mod': 4, 'Maj1': 5, 'Maj2': 6, 'Sev': 7, 'Sal': 8},
        'GarageFinish': {'Fin': 1, 'RFn': 2, 'Unf': 3, 'NA': 4},
        'PavedDrive': {'Y': 1, 'P': 2, 'N': 3},
        'PoolQC': {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'NA': 5},
        'Fence': {'GdPrv': 1, 'GdWo': 2, 'NA': 3, 'MnWw': 4, 'MnPrv': 5},
        'MiscFeature': {'Elev': 1, 'Gar2': 2, 'Othr': 3, 'Shed': 4, 'TenC': 5, 'NA': 6},
        'SaleType': {'WD': 1, 'CWD': 2, 'VWD': 3, 'New': 4, 'COD': 5, 'Con': 6, 'ConLw': 7, 'ConLI': 8, 'ConLD': 9, 'Oth': 10},
        'SaleCondition': {'Normal': 1, 'Abnorml': 2, 'AdjLand': 3, 'Alloca': 4, 'Family': 5, 'Partial': 6}
    }

    # Anwenden der Mapping-Dictionaries auf die entsprechenden Spalten
    for column, mapping in mappings.items():
        df_train[column] = df_train[column].map(mapping)
        df_test[column] = df_test[column].map(mapping)

    # One-Hot Encoding ohne Ordnung
    categorical_columns = [
        'MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'RoofStyle', 
        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 
        'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]
    df_train = pd.get_dummies(df_train, columns=categorical_columns, prefix=categorical_columns)
    df_test = pd.get_dummies(df_test, columns=categorical_columns, prefix=categorical_columns)

     # Entfernen von 'FireplaceQu'-Spalte
    if 'FireplaceQu' in df_train.columns:
        df_train.drop(columns=['FireplaceQu'], inplace=True)
    if 'FireplaceQu' in df_test.columns:
        df_test.drop(columns=['FireplaceQu'], inplace=True)

    # Spalten in den Testdaten anpassen
    missing_columns = set(df_train.columns) - set(df_test.columns)
    for column in missing_columns:
        df_test[column] = 0  # Füge fehlende Spalten hinzu und setze ihre Werte auf 0
    
    # Spalten in den Traindaten anpassen
    missing_columns = set(df_test.columns) - set(df_train.columns)
    for column in missing_columns:
        df_train[column] = 0  # Füge fehlende Spalten hinzu und setze ihre Werte auf 0

     # Entfernen von Spalten mit mehr als 50% Fehlwerten
    threshold = len(df_train) * 0.5
    df_train.dropna(thresh=threshold, axis=1, inplace=True)
    df_test.dropna(thresh=threshold, axis=1, inplace=True)

    # Fehlende Werte mit dem Spaltenmittelwert ersetzen
    df_train.fillna(df_train.mean(), inplace=True)
    df_test.fillna(df_test.mean(), inplace=True)

    # Erstelle eine Liste der Spaltennamen in der richtigen Reihenfolge
    column_order = df_train.columns.tolist()

    # Wähle nur die relevanten Spalten in den Testdaten aus und ordne sie nach der Reihenfolge der Trainingsdaten
    df_test = df_test[column_order]

    return df_train, df_test

def data_correlation(df):
    correlation_matrix = df.corr()
    correlation_with_saleprice = correlation_matrix['SalePrice'].sort_values(ascending=False)

    # Die zehn Merkmale mit den höchsten positiven Korrelationen
    top_positive_correlations = correlation_with_saleprice[1:11]

    # Die zehn Merkmale mit den höchsten negativen Korrelationen
    top_negative_correlations = correlation_with_saleprice.tail(10)
    extended_df = pd.concat([top_positive_correlations, top_negative_correlations], axis=0)

    return correlation_with_saleprice,  extended_df


def data_modeling(df_train,df_test):
    X = df_train.drop(columns=['SalePrice'])  # Features
    y = df_train['SalePrice']  # Target variable
    X_test = df_test.drop(columns=['SalePrice'])
    # Initialisierung der Modelle
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor()
    gradient_boosting_model = GradientBoostingRegressor()

    # Anpassen der Modelle an die Trainingsdaten
    linear_model.fit(X, y)
    random_forest_model.fit(X, y)
    gradient_boosting_model.fit(X, y)

    from sklearn.model_selection import cross_val_score

    # Kreuzvalidierung für lineare Regression
    linear_scores = cross_val_score(linear_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    #print("Linear Regression Cross-Validation Scores:")
    #print(linear_scores)

    # Kreuzvalidierung für Random Forest
    random_forest_scores = cross_val_score(random_forest_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    #print("\nRandom Forest Cross-Validation Scores:")
    #print(random_forest_scores)

    # Kreuzvalidierung für Gradient Boosting
    gradient_boosting_scores = cross_val_score(gradient_boosting_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    #print("\nGradient Boosting Cross-Validation Scores:")
    #print(gradient_boosting_scores)

    # Vorhersagen für lineare Regression
    linear_predictions = linear_model.predict(X_test)

    # Vorhersagen für Random Forest
    random_forest_predictions = random_forest_model.predict(X_test)

    # Vorhersagen für Gradient Boosting
    gradient_boosting_predictions = gradient_boosting_model.predict(X_test)

    # Rückgabe der Vorhersagen in einem DataFrame
    result_df = pd.DataFrame({
        'ID': df_test['Id'],
        'LinearRegression': linear_predictions,
        'RandomForest': random_forest_predictions,
        'GradientBoosting': gradient_boosting_predictions
    })

    return result_df


[df_train1, df_test1]=data_preparation(df_train, df_test)
df_final_result=data_modeling(df_train1,df_test1)
# Exportiere die Ergebnisse der linearen Regression in eine CSV-Datei
df_final_result[['ID', 'LinearRegression']].to_csv('linear_regression_predictions.csv', index=False, header=['Id', 'SalePrice'])

# Exportiere die Ergebnisse des Random Forest in eine CSV-Datei
df_final_result[['ID', 'RandomForest']].to_csv('random_forest_predictions.csv', index=False, header=['Id', 'SalePrice'])

# Exportiere die Ergebnisse des Gradient Boosting in eine CSV-Datei
df_final_result[['ID', 'GradientBoosting']].to_csv('gradient_boosting_predictions.csv', index=False, header=['Id', 'SalePrice'])