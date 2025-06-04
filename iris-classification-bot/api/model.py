import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    feature_info = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'feature_ranges': {
            'sepal_length': (min(iris.data[:,0]), max(iris.data[:,0])),
            'sepal_width': (min(iris.data[:,1]), max(iris.data[:,1])),
            'petal_length': (min(iris.data[:,2]), max(iris.data[:,2])),
            'petal_width': (min(iris.data[:,3]), max(iris.data[:,3])),
        }
    }

    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    print("Model and feature info saved")

if __name__ == '__main__':
    train_and_save_model()
