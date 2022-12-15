# read data
df = pd.read_pickle('.\\italianVAT_dataset')

# list of classifiers and embeddings to try
classifiers = ['linearsvc', 'randomforest', 'gaussiannb', 'kneighbors', 'svc']
# classifiers = ['random', 'majority']
embeddings = ['tfidf', 'sbert']

# outcome prediction
predict_outcome(df, classifiers, embeddings)

# add mot
optionals = ['mot']
predict_outcome(df, classifiers, embeddings, optionals)

# add dec
optionals = ['dec']
predict_outcome(df, classifiers, embeddings, optionals)

# add mot and dec
optionals = ['mot', 'dec']
predict_outcome(df, classifiers, embeddings, optionals)