import pandas as pd
import numpy as np

import pickle
from sklearn.preprocessing import LabelEncoder

import os

# ===== Načítanie parametrov z model_param_file.txt =====

param_file = 'model_param_file.txt'  # predpokladáme, že je v rovnakom adresári ako tento skript

params = {}
with open(param_file, 'r') as f:
    for line in f:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            params[key.strip()] = value.strip()

# ===== Načítanie modelu =====
with open(params['model_path'], 'rb') as file:
    final_model = pickle.load(file)

# ===== Načítanie features =====
with open(params['features_path'], 'rb') as f:
    features_train = pickle.load(f)

# ===== Načítanie LabelEncodera =====
with open(params['encoder_path'], 'rb') as f:
    encoder = pickle.load(f)

# ===== Načítanie datasetu =====
data_prep_1 = pd.read_csv(params['data_path'], delimiter=";")

print("Všetky súbory úspešne načítané.")


###############


data_prep_1 = data_prep_1.drop(['uzavreni_datum_cas','ucetni_datum', 'rozhodne_datum'], axis=1)

data_prep_1 = data_prep_1.drop(['trvani', 'doba_otevreni_uctenky_sec'], axis=1)


# VYMAZ STLCPOV KVOLI DATALEAKGE
data_prep_1 = data_prep_1.drop(['UCTENKA_CENA_VCETNE_DPH_PO_SLEVE','kod_uctenky' ], axis=1)

data_prep_1 = data_prep_1.drop(['produkt_nazev.1', 'restaurace_nazev', 'restaurant_id', 'produktova_rada','kategorie_segmentu'], axis=1)



# Ostranenie riadkov kde je produkt_id null , budeme ho predikovať, v zasade ide o rozne kupony a akcie

data_prep_2 = data_prep_1[data_prep_1['produkt_id'].notnull()]


# Odstranenie zaznamov jde je typ objednávky iný ako "tady/sebou"


data_prep_3 = data_prep_2[data_prep_2['typ_objednavky'].isin(['tady', 'sebou'])]



# ODSTRANENIE produktov typu "akčni xxx menu" duplicitny zaznam hovori len o tom že bageta je akčni a tuto informáciu


# Make explicit copy to avoid chained-assignment warnings
data_prep_4 = data_prep_3.copy()

# Vytvorenie príznakov pre rôzne podmienky
data_prep_4['is_akcni_menu'] = data_prep_4['produkt_nazev'].str.startswith('Akční', na=False) & data_prep_4['produkt_nazev'].str.endswith('menu', na=False)
data_prep_4['is_menu'] = data_prep_4['produkt_nazev'].str.startswith('MENU', na=False)
data_prep_4['is_superpytlik'] = data_prep_4['produkt_nazev'].str.contains('Superpytlík', na=False)
data_prep_4['is_box_pro_2'] = data_prep_4['produkt_nazev'].str.startswith('Box pro 2', na=False)
data_prep_4['is_Snidane'] = data_prep_4['produkt_nazev'].str.startswith('SNÍDANĚ', na=False)

# Vytvorenie špeciálnych príznakov pre každý doklad_id
data_prep_4['has_akcni_menu'] = data_prep_4.groupby('doklad_id')['is_akcni_menu'].transform('max').astype(int)
data_prep_4['has_menu'] = data_prep_4.groupby('doklad_id')['is_menu'].transform('max').astype(int)
data_prep_4['has_superpytlik'] = data_prep_4.groupby('doklad_id')['is_superpytlik'].transform('max').astype(int)
data_prep_4['has_box_pro_2'] = data_prep_4.groupby('doklad_id')['is_box_pro_2'].transform('max').astype(int)
data_prep_4['has_Snidane'] = data_prep_4.groupby('doklad_id')['is_Snidane'].transform('max').astype(int)

# Odstránenie pomocných stĺpcov (now safe)
data_prep_4.drop(columns=['is_akcni_menu', 'is_menu', 'is_superpytlik', 'is_box_pro_2', 'is_Snidane'], inplace=True)


#####################################################################################################



# Identifikácia záznamov, ktoré splňujú dané podmienky
mask_to_remove = (
    data_prep_4['produkt_nazev'].str.startswith('Akční', na=False) & data_prep_4['produkt_nazev'].str.endswith('menu', na=False)
) | (
    data_prep_4['produkt_nazev'].str.startswith('MENU', na=False)
) | (
    data_prep_4['produkt_nazev'].str.contains('Superpytlík', na=False)
) | (
    data_prep_4['produkt_nazev'].str.startswith('Box pro 2', na=False)
) | (
    data_prep_4['produkt_nazev'].str.startswith('SNÍDANĚ', na=False)
)

# Odstránenie záznamov na základe masky
data_prep_4_cleaned = data_prep_4[~mask_to_remove].copy()


#kontrola pri probléme
# Výstup
# print(f"Počet záznamov pred odstránením: {len(data_prep_4)}")
# print(f"Počet záznamov po odstránení: {len(data_prep_4_cleaned)}")


##################################



data_prep_4 = data_prep_4_cleaned


#doklad_id na int datatype

doklad_id_int = pd.DataFrame(
    enumerate(data_prep_4['doklad_id'].unique(), start=1)
).rename({0: 'doklad_id_int', 1: 'doklad_id'}, axis=1)

# Pridanie stĺpca 'doklad_id_int' do hlavného dataframe
data_prep_4 = data_prep_4.merge(doklad_id_int, on='doklad_id', how='left')

# Teraz už môžeš bezpečne konvertovať na integer
data_prep_4['doklad_id_int'] = data_prep_4['doklad_id_int'].astype(int)

########

produkt_id_int = pd.DataFrame(
    enumerate(data_prep_4['produkt_id'].unique(), start=1)
).rename({0: 'produkt_id_int', 1: 'produkt_id'}, axis=1)

# Spojenie s hlavným dataframe 
data_prep_4 = data_prep_4.merge(produkt_id_int, on='produkt_id', how='left')

# Pridanie stĺpca
data_prep_4['produkt_id_int'] = data_prep_4['produkt_id_int'].astype(int)


#Zoradenie podľa case otvorenia objednávky
data_prep_4 = data_prep_4.sort_values(by=["otevreni_datum_cas"])



######

# zmena hodnot: ANO/NE na 1/0 , vytvorenie funkcie

def map_values_to_boolean(df, mapping_dict):
  """
  Maps specific values in a DataFrame to boolean values.

  Args:
    df: The pandas DataFrame.
    mapping_dict: A dictionary mapping values to boolean values.

  Returns:
    A new DataFrame with mapped boolean values.
  """

  for col in df.columns:
    df[col] = df[col].map(mapping_dict).fillna(df[col])

  return df


mapping_dict = {'ne': 0, 'ano': 1}

#####

# použitie funkcie
data_prep_4 = map_values_to_boolean(data_prep_4, mapping_dict)


# zmena datumu na datetime

data_prep_4['otevreni_datum_cas'] = pd.to_datetime(data_prep_4['otevreni_datum_cas'])


###########

# pridanie feature mesiac, rok,  ako int

data_prep_4['Month_Number'] = data_prep_4['otevreni_datum_cas'].dt.month



data_prep_4['Year_Number'] = data_prep_4['otevreni_datum_cas'].dt.year






# Vymazanie riadkov, kde is_pecivo nie je 1,
# jedna sa o položky vyslovene aku pečivo bolo vybrane k bagete, tato informacia je už v sltpci "typ_pečiva"

data_prep_4 = data_prep_4[data_prep_4['is_pecivo'] != 1]


data_prep_4['typ_peciva'] = data_prep_4['typ_peciva'].fillna('--empty--')

#########


data_prep_4['doklad_id_int'] = data_prep_4['doklad_id_int'].astype(int)


data_prep_4['produkt_id_int'] = data_prep_4['produkt_id_int'].astype(int)

data_prep_4['poradi'] = data_prep_4['poradi'].astype(int)


####################################################################



column_names_dumm = ['kasa', 'typ_kasy', 'typ_objednavky', 'typ_peciva', 'food_drink', 'segment']


data_prep_4 = pd.get_dummies(data_prep_4, columns = column_names_dumm)

###############

product_counts = data_prep_4['produkt_nazev'].value_counts()

# Filtrovanie produktov s počtom >= 200 , odkomentovanie nutné len pre tréning modelu a vytváranie súboru pre tréning, inak je nutné mať zakomentovane
#data_prep_4 = data_prep_4[data_prep_4['produkt_nazev'].isin(product_counts[product_counts >= 200].index)]


# odstranenie prepitné ako položky
data_prep_4 = data_prep_4[~data_prep_4['produkt_nazev'].isin(['Tip for staff', 'Tip for restaurant'])]



############### Zoradenie položiek
df = data_prep_4

df = df.sort_values(by=['doklad_id_int', 'poradi'])


# Vytvorenie sekvencie položiek v rámci každej objednávky
df['item_sequence'] = df.groupby('doklad_id_int').cumcount() + 1

# Výpočet kumulatívnej hodnoty objednávky pre každú položku
df['cumulative_order_value'] = df.groupby('doklad_id')['POLOZKA_CENA_VCETNE_DPH_PO_SLEVE'].cumsum()


# nova premenná ktorá hovorí či bol produkt_id už v objednávke


unique_products = df['produkt_nazev'].unique()

# Inicializácia stĺpcov s hodnotami 0
for product in unique_products:
    df[f'product_{product}_before'] = 0

# Funkcia na nastavenie hodnôt pre každý riadok
def set_previous_products(group):
    previous = set()
    for index, row in group.iterrows():
        for product in unique_products:
            if product in previous:
                group.at[index, f'product_{product}_before'] = 1
        previous.add(row['produkt_nazev'])
    return group

# Aplikácia funkcie na každý doklad
df = df.groupby('doklad_id', group_keys=False).apply(set_previous_products)


###################### 
# odkomentovať pri tréningu modelu, vytvara pickle dataframe na ktorom je trénovaný model
#df.to_pickle('C:\\diplomka_work\\python_code\\data_pickles\\train_cely.pkl')

column_names_dumm = ['produkt_nazev']
df = pd.get_dummies(df, columns = column_names_dumm)


# --- Zarovnanie stĺpcov ---
missing_cols = set(features_train) - set(df.columns)
for col in missing_cols:
    df[col] = 0  # Doplníme chýbajúce stĺpce nulami

# Odstrániť stĺpce navyše
df = df[features_train]

final_order_predict = final_model.predict(df)


predicted_labels = encoder.inverse_transform(final_order_predict)

print("Predikované kategórie:", predicted_labels)


predicted_output_file = params['predicted_output_path']

with open(predicted_output_file, 'w', encoding='utf-8') as f:
    for label in predicted_labels:
        f.write(f"{label}\n")

print(f"Predikcie uložené do {predicted_output_file}")








