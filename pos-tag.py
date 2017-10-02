from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

X = []
y = []
f = open('id-ud-train.conllu', 'r')
for line in f:
  sentence = line.split("\t")
  if sentence[0].isdigit():
    X.append( {
      'word': sentence[1],
      'is_first': sentence[0] == '1',
      'is_last': sentence[0] == len(sentence) - 1,
      'is_capitalized': sentence[1][0].upper() == sentence[1][0],
      'is_all_caps': sentence[1].upper() == sentence[1],
      'is_all_lower': sentence[1].lower() == sentence[1],
      'prefix-1': sentence[1][0],
      'prefix-2': sentence[1][:2],
      'prefix-3': sentence[1][:3],
      'suffix-1': sentence[1][-1],
      'suffix-2': sentence[1][-2:],
      'suffix-3': sentence[1][-3:],
      'prev_tag': '' if sentence[0] == '1' else y[len(X)-1],
      'prev_word': '' if sentence[0] == '1' else X[len(X)-1]['word'],
      'has_hyphen': '-' in sentence[1],
      'is_numeric': sentence[1].isdigit(),
      'capitals_inside': sentence[1][1:].lower() != sentence[1][1:] }
    )
    y.append(sentence[3])


print("There is " + str(len(X)) + " Feature data")
vec = DictVectorizer()
X = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.75, random_state=50)

print("Start the training")
clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,2))
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print("Score is ")
print(clf.score(X_test,y_test))


