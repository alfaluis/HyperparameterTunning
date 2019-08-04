import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,
import pandas as pd


root = os.path.join(os.getcwd(), 'database')

df = pd.read_csv(os.path.join(root, 'base_churn_aug18_version_04_02_2019.csv'), sep=',')
df.set_index('RUTNUM', drop=True, inplace=True)
features_cols = [col for col in df.columns if 'FUGA' not in col]
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, features_cols], df.loc[:, 'FUGA'],
                                                    test_size=0.3, random_state=42)

std_scaler = StandardScaler().fit(X=X_train)
x_train_norm = pd.DataFrame(data=std_scaler.transform(X_train), columns=features_cols)
x_test_norm = pd.DataFrame(data=std_scaler.transform(X_test), columns=features_cols)

# Create a classifier: a support vector classifier
kernel_svm = svm.SVC(gamma=.2)
linear_svm = svm.LinearSVC(class_weight='balanced')

# create pipeline from kernel approximation
# and linear svm
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)

feature_map_nystroem = Nystroem(gamma=.2, n_components=500, random_state=1)
list_cm = list()
for n_samples in range(500, 5001, 500):
    print('Approximation with {} samples'.format(n_samples))
    feature_map_nystroem.fit(x_train_norm.iloc[:n_samples])
    data_transformed = feature_map_nystroem.transform(x_train_norm)
    linear_svm.fit(X=data_transformed, y=y_train)
    pred = linear_svm.predict(X=feature_map_nystroem.transform(x_test_norm))
    list_cm.append(confusion_matrix(y_true=y_test, y_pred=pred))
    print('CM', list_cm[-1])

pred = linear_svm.predict(X=feature_map_nystroem.transform(x_test_norm))
score = linear_svm.score(X=feature_map_nystroem.transform(x_test_norm), y=y_test)
cm = confusion_matrix(y_true=y_test, y_pred=pred)
print('Score:', score)
print('CM:', cm)
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", svm.LinearSVC())])

nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                         ("svm", svm.LinearSVC())])

# fit and predict using linear and kernel svm:
kernel_svm.fit(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)

linear_svm.fit(data_train, targets_train)
linear_svm_score = linear_svm.score(data_test, targets_test)

sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []

for D in sample_sizes:
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)

    nystroem_approx_svm.fit(data_train, targets_train)
    fourier_approx_svm.fit(data_train, targets_train)

    fourier_score = fourier_approx_svm.score(data_test, targets_test)
    nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)

# plot the results:
accuracy = plt.figure()
accuracy = plt.subplot()
accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_svm_score, linear_svm_score], label="linear svm")
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [kernel_svm_score, kernel_svm_score], label="rbf svm")

# legends and labels
accuracy.set_title("Classification accuracy")
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
accuracy.set_ylim(np.min(fourier_scores), 1)
accuracy.set_ylabel("Classification accuracy")
accuracy.legend(loc='best')