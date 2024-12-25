import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import export_text

# Load the model and metadata
model_data = joblib.load("decision_tree_model.pkl")
model = model_data["model"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]

# Streamlit app
st.title("Decision Tree Classifier App")
st.write("Enter the feature values to get the classification result:")

# Dynamically create input fields for each feature
inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, max_value=10.0, value=5.0)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    # Prepare the input as a DataFrame with feature names
    input_features = pd.DataFrame([inputs], columns=feature_names)
    
    # Predict the class
    prediction = model.predict(input_features)[0]
    predicted_class_name = target_names[prediction]
    
    # Trace the decision path
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    
    # Get the decision path
    node_indicator = model.decision_path(input_features)
    leave_id = model.apply(input_features)
    
    st.write(f"Predicted Class: {predicted_class_name}")
    
    # Display the decision path
    st.subheader("Decision Path:")
    path_str = ""
    for node_id in node_indicator.indices:
        if leave_id[0] == node_id:
            path_str += f"Reached leaf node {node_id}: Predict class {predicted_class_name}\n"
        else:
            if input_features.iloc[0, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            path_str += (
                f"Node {node_id}: (Feature '{feature_names[feature[node_id]]}' "
                f"is {threshold_sign} {threshold[node_id]:.2f})\n"
            )
    
    # Render the path in a code block
    st.code(path_str)

    # Export and visualize the tree
    st.subheader("Decision Tree Visualization:")
    st.graphviz_chart(export_text(model, feature_names=feature_names))
