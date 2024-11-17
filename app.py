from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app, resources={r"/recommend/*": {"origins": "https://lucent-muffin-530d09.netlify.app"}})  # Allow your frontend origin

# Sample DataFrame (Replace with your actual dataset)
df = pd.read_csv('datasets/amazon.csv')  # Replace with your dataset file path

# Preprocessing: Fill missing values if necessary
df['category'] = df['category'].fillna('')
df['product_name'] = df['product_name'].fillna('')
df['about_product'] = df['about_product'].fillna('')

# Combine features to create a description for each product
df['product_description'] = df['category'] + ' ' + df['product_name'] + ' ' + df['about_product']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the product descriptions
tfidf_matrix = tfidf.fit_transform(df['product_description'])

# Compute cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend products
def recommend_products(product_id, top_n):
    try:
        # Ensure product_id is a string and match it with the dataframe
        idx = df[df['product_id'] == product_id].index[0]

        # Get pairwise similarity scores for this product
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the products based on similarity scores (in descending order)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top_n most similar products
        sim_scores = sim_scores[1:top_n+1]  # Skip the first one because it's the product itself
        product_indices = [i[0] for i in sim_scores]

        # Return top_n recommended products
        recommended_products = df.iloc[product_indices]
        return recommended_products[['product_id', 'product_name', 'category', 'discounted_price', 'rating']].to_dict(orient='records')

    except Exception as e:
        return {"error": str(e)}

# API endpoint for product recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id', type=str)  # Ensure product_id is treated as a string
    num_of_products = request.args.get('num_of_products', type=int, default=5)  # Default is 5, but dynamic

    if not product_id:
        return jsonify({"error": "Product ID is required"}), 400

    recommended_products = recommend_products(product_id, num_of_products)

    if "error" in recommended_products:
        return jsonify(recommended_products), 500

    return jsonify(recommended_products)

if __name__ == '__main__':
    app.run(debug=True)
