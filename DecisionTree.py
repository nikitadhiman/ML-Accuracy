import pandas as pd
import time
from sklearn.metrics import accuracy_score


# Set timer
start_time = time.time()

# Load the dataset
file_path = './Fraud_check.csv'  # Update this path as necessary
fraud_data = pd.read_csv(file_path)

# Prepare the dataset by creating a new target column 'Risk' based on 'Taxable.Income'
fraud_data['Risk'] = fraud_data['Taxable.Income'].apply(lambda x: 0 if x <= 30000 else 1)

# Drop the 'City.Population' and 'Taxable.Income' columns
#fraud_data_cleaned = fraud_data.drop(columns=['City.Population', 'Taxable.Income'])

# Convert categorical columns into numeric using one-hot encoding
fraud_data_encoded = pd.get_dummies(fraud_data, columns=['Undergrad', 'Marital.Status', 'Urban'], drop_first=True)

# Split the data into features (X) and target (y)
X = fraud_data_encoded.drop(columns=['Risk'])
y = fraud_data_encoded['Risk']

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Implementing a basic Decision Tree Classifier using GINI index (without any existing libraries)
class DecisionTree:
    def __init__(self):
        self.tree = None
    
    def gini(self, groups, classes):
        # Calculate Gini Index for a split dataset
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                proportion = [row[-1] for row in group].count(class_val) / size
                score += proportion * proportion
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def split(self, index, value, dataset):
        # Split a dataset into two based on an attribute and its value
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def best_split(self, dataset):
        # Find the best split point for a dataset
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.split(index, row[index], dataset)
                gini = self.gini(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group):
        # Create a terminal node value
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split_node(self, node, max_depth, min_size, depth):
        # Recursive splitting of nodes
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.best_split(left)
            self.split_node(node['left'], max_depth, min_size, depth+1)
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.best_split(right)
            self.split_node(node['right'], max_depth, min_size, depth+1)

    def build_tree(self, train, max_depth, min_size):
        # Build the decision tree
        root = self.best_split(train)
        self.split_node(root, max_depth, min_size, 1)
        return root

    def fit(self, X, y, max_depth=5, min_size=10):
        # Train the decision tree on training data
        dataset = [list(X.iloc[i]) + [y.iloc[i]] for i in range(len(X))]
        self.tree = self.build_tree(dataset, max_depth, min_size)

    def predict_row(self, node, row):
        # Make a prediction for a single row of data
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']

    def predict(self, X):
        # Make predictions for a dataset
        predictions = [self.predict_row(self.tree, list(X.iloc[i])) for i in range(len(X))]
        return predictions

# Convert the training and test sets into the proper format
X_train_list = X_train.reset_index(drop=True)
X_test_list = X_test.reset_index(drop=True)

# Initialize and train the Decision Tree
decision_tree = DecisionTree()
decision_tree.fit(X_train_list, y_train)

# Make predictions on the test set
predictions = decision_tree.predict(X_test_list)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)
execution_time = time.time() - start_time
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f"Execution Time: {execution_time:.2f} seconds")

