import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 501)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot  as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("diabetes.csv")
data = data.copy()

df = data.copy()
df.head()
df

# Menghapus kolom 'Pregnancies'
df = pd.DataFrame(data.drop('Pregnancies', axis=1))
df

# Mendeskripsikan DataFrame
df.describe(include='all')

# Informasi DataFrame
df.info()

# Normalisasi data
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

# Membagi Data menjadi Fitur dan Target
X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

# Membagi Data menjadI Data Latih dan Data Uji
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

# Membuat dan Melatih Model
model = GaussianNB()
model.fit(X_train, Y_train.ravel())

# Memprediksi Data uji
y_pred = model.predict(X_test)

# Menghitung dan Menampilkan akurasi
from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

## CLEARING DATA GLUCOSE NULL

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["Glucose"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between Glucose & Diabetes (the density is visible)" ,fontsize = 20)
plt.xticks (range (0 , 205 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('Glucose', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df = df[df['Glucose'] >= 10]
df

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["Glucose"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between Glucose & Diabetes (the density is visible & noises have been removed)" ,fontsize = 20)
plt.xticks (range (0 , 205 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('Glucose', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df.reset_index(drop=True, inplace=True) # reseting index
df

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

## CLEARING DATA BLOOD PREASSURE NULL

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["BloodPressure"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between BloodPressure & Diabetes (the density is visible)", fontsize = 20)
plt.xticks (range (0 , 140 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('BloodPressure', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df = df[df['BloodPressure'] >= 10]
df

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["BloodPressure"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between BloodPressure & Diabetes (the density is visible & noises have been removed)" ,fontsize = 20)
plt.xticks (range (0 , 120 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('BloodPressure', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df.reset_index(drop=True, inplace=True) # reseting index
df

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

## CLEARING DATA SKIN TICKNESS NULL

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["SkinThickness"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between SkinThickness & Diabetes (the density is visible)", fontsize = 20)
plt.xticks (range (0 , 120 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('SkinThickness', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df = df[df['SkinThickness'] >= 1]
df

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["SkinThickness"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between SkinThickness & Diabetes (the density is visible & noises have been removed)" ,fontsize = 20)
plt.xticks (range (0 , 110 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('SkinThickness', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df.reset_index(drop=True, inplace=True) # reseting index
df

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

## CLEARING DATA INSULIN NULL

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["Insulin"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between Insulin & Diabetes (the density is visible)" ,fontsize = 20)
plt.xticks (range (0 , 1000 , 100), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('Insulin', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df = df[df['Insulin'] >= 1]
df

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["Insulin"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between Insulin & Diabetes (the density is visible & noises have been removed)" ,fontsize = 20)
plt.xticks (range (0 , 1000 , 100), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('Insulin', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df.reset_index(drop=True, inplace=True) # reseting index
df

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

## CLEARING DATA BMI NULL

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["BMI"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between BMI & Diabetes (the density is visible)" ,fontsize = 20)
plt.xticks (range (0 , 70 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('BMI', fontsize = 20 )
plt.grid ()
plt.show ()

df = df[df['BMI'] >= 1]
df

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["Age"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between Age & Diabetes (the density is visible)" ,fontsize = 20)
plt.xticks (range (0 , 100 , 10), fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('Age', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df.reset_index(drop=True, inplace=True) # reseting index
df

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

## CLEARING DATA DIABETES PEDIGREE FUNCTION NULL

plt.figure(figsize = [20, 4] , dpi = 150)
plt.scatter (df["DiabetesPedigreeFunction"] , df["Outcome"] , color = "sienna")
plt.title ("Relationship between DiabetesPedigreeFunction & Diabetes (the density is visible)" ,fontsize = 20)
plt.xticks (fontsize = 20)
plt.yticks (fontsize = 20)
plt.xlabel ('DiabetesPedigreeFunction', fontsize = 20 )
plt.ylabel ('Diabetes' , fontsize = 20)
plt.grid ()
plt.show ()

df.describe()

df.reset_index(drop=True, inplace=True) # reseting index
df

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dff = scaler.fit_transform(df)
dff = pd.DataFrame(dff, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])

X = pd.DataFrame(dff.drop('Outcome', axis=1))
Y = dff['Outcome'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
print("\33[43m Accuracy Is:", metrics.accuracy_score(Y_test, y_pred))

n = df.groupby('Outcome')[['Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].mean()
ax = n.plot(kind='bar', figsize=(15, 10), cmap='BrBG')
plt.title("influence of each column on Outcome")
plt.xlabel('Outcome')
plt.ylabel('Average')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                xytext=(0, 9),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=10, color='black')

plt.show()

columns = ['Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
palette ="copper"
for column in columns:
    plt.figure(figsize=(15,2))
    sns.boxplot(x=df[column], palette=palette)
    plt.title(column)
    stats = df[column].describe()
    stats_text = ", ".join([f"{key}: {value:.2f}" for key, value in stats.items()])
    print(f"\n{column} Statistics:\n{stats_text}")
    plt.show()

glucose_bins=pd.cut(df["Glucose"],bins=[0,40,90,130,200],labels=["0-40","40-90","90-130","130-200"])
plt.figure(figsize=(20,10))
ax = sns.countplot(x=glucose_bins, data=df, hue="Outcome", palette='copper')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height, int(height), ha="center", va="bottom", fontsize=12)
plt.show()

insulin_bins=pd.cut(df["Insulin"],bins=[0,50,100,150,200,1000],labels=["0-50","50-100","100-150","150-200",">200"])
plt.figure(figsize=(20,10))
ax = sns.countplot(x=insulin_bins, data=df, hue="Outcome", palette='copper')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height, int(height), ha="center", va="bottom", fontsize=12)
plt.show()

BP_bins=pd.cut(df["BloodPressure"],bins=[0,50,80,100,200],labels=["0-50","50-80","80-100","100-200"])
plt.figure(figsize=(20,10))
ax = sns.countplot(x=BP_bins, data=df, hue="Outcome", palette='copper')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height, int(height), ha="center", va="bottom", fontsize=12)
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
model.classes_

plt.figure(figsize=(15, 8))
sns.set(font_scale=1.2)
sns.heatmap(confusion_matrix(Y, model.predict(X)) , annot=True, fmt="d", cmap="BrBG", cbar=False,
            xticklabels=['Normal', 'Diabetes'], yticklabels=['Normal', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(Y, model.predict(X)))

X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Transformasi hanya pada fitur
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Buat DataFrame baru dengan fitur yang telah di-scale
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

# Latih model
model = GaussianNB()
model.fit(X_train, Y_train)

def get_user_input():
    print("Silakan Masukan Nilai Setiap Variabel Berikut:")
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("BloodPressure: "))
    skin_thickness = float(input("SkinThickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("DiabetesPedigreeFunction: "))
    age = float(input("Age: "))
    return [[glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]

# Fungsi untuk membuat diagram lingkaran
def plot_pie_chart(probability):
    labels = ['Diabetes', 'Tidak Diabetes']
    sizes = [probability, 1 - probability]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # Mencuatkan bagian 'Diabetes'

    plt.figure(figsize=(3, 3))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('')
    plt.axis('equal')  # Memastikan lingkaran tidak terdistorsi
    plt.show()

def run_prediction():
    while True:
        # Ambil input dari user
        new_input = get_user_input()

        # Transformasi input baru menggunakan scaler
        new_input_scaled = scaler.transform(new_input)

        # Prediksi probabilitas untuk input baru
        newf_proba = model.predict_proba(new_input_scaled)[0][1]

        # Output prediksi dan rating probabilitas
        print(f"==================================")

        if newf_proba < 0.4:
            print("Hasil Prediksi: TIDAK DIABETES")
        elif 0.4 <= newf_proba < 0.5:
            print("Hasil Prediksi: KEMUNGKINAN TIDAK DIABETES")
        elif 0.5 <= newf_proba < 0.6:
            print("Hasil Prediksi: KEMUNGKINAN DIABETES")
        else:
            print("Hasil Prediksi: DIABETES")

        # Tambahkan Rating Prediksi
        print(f"Rating Prediksi: {newf_proba:.2f}")
        print(f"==================================")

        # Tampilkan diagram lingkaran
        plot_pie_chart(newf_proba)

        # Tanya user apakah ingin mengulang pengujian
        print("\nApakah akan melakukan pengujian kembali?")
        print("1. Ya")
        print("2. Tidak")
        pilihan = input("Masukkan pilihan Anda (1 atau 2): ")

        if pilihan == "1":
            print("\nMemulai pengujian ulang...\n")
        elif pilihan == "2":
            print("TERIMA KASIH TELAH MENCOBA")
            break
        else:
            print("Pilihan tidak valid. Program akan berhenti.")
            break

# Jalankan program
run_prediction()
