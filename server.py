from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@dataclass
class Recipe:
    name: str
    ingredients: List[str]
    recipe_category: Optional[str] = None
    recipe_servings: Optional[int] = None
    calories: Optional[float] = None
    fat_content: Optional[float] = None
    saturated_fat_content: Optional[float] = None
    cholesterol_content: Optional[float] = None
    sodium_content: Optional[float] = None
    carbohydrate_content: Optional[float] = None
    fiber_content: Optional[float] = None
    sugar_content: Optional[float] = None
    protein_content: Optional[float] = None
    prep_time: Optional[float] = None
    cook_time: Optional[float] = None
    total_time: Optional[float] = None
    description: Optional[str] = None
    instructions: Optional[str] = None


model = None
scaler = None
feature_columns = None
ingredient_encoder = None
category_encoder = None


def load_model():
    global model, scaler, feature_columns, ingredient_encoder, category_encoder
    
    model_path = 'xgb_model_bo2.pkl'
    scaler_path = 'scaler_bo2.pkl'
    feature_columns_path = 'feature_columns_bo2.pkl'
    ingredient_encoder_path = 'ingredient_encoder_bo2.pkl'
    category_encoder_path = 'category_encoder_bo2.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using placeholder model.")
        print("To use the actual model, save it from the notebook using:")
        print("import pickle")
        print("with open('xgb_model_bo2.pkl', 'wb') as f:")
        print("    pickle.dump(xgb_best, f)")
        model = XGBClassifier(random_state=42, reg_alpha=1, reg_lambda=1)
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_path}")
    else:
        scaler = None
        print(f"Scaler file {scaler_path} not found. Scaling will be skipped.")
    
    if os.path.exists(feature_columns_path):
        with open(feature_columns_path, 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"Feature columns loaded from {feature_columns_path}")
    else:
        feature_columns = None
        print(f"Feature columns file {feature_columns_path} not found.")
    
    if os.path.exists(ingredient_encoder_path):
        with open(ingredient_encoder_path, 'rb') as f:
            ingredient_encoder = pickle.load(f)
        print(f"Ingredient encoder loaded from {ingredient_encoder_path}")
    else:
        ingredient_encoder = None
        print(f"Ingredient encoder file {ingredient_encoder_path} not found.")
    
    if os.path.exists(category_encoder_path):
        with open(category_encoder_path, 'rb') as f:
            category_encoder = pickle.load(f)
        print(f"Category encoder loaded from {category_encoder_path}")
    else:
        category_encoder = None
        print(f"Category encoder file {category_encoder_path} not found.")


def prepare_features(recipe: Recipe) -> pd.DataFrame:
    features = {}
    
    features['Calories'] = recipe.calories if recipe.calories is not None else 0.0
    features['FatContent'] = recipe.fat_content if recipe.fat_content is not None else 0.0
    features['SaturatedFatContent'] = recipe.saturated_fat_content if recipe.saturated_fat_content is not None else 0.0
    features['CholesterolContent'] = recipe.cholesterol_content if recipe.cholesterol_content is not None else 0.0
    features['SodiumContent'] = recipe.sodium_content if recipe.sodium_content is not None else 0.0
    features['CarbohydrateContent'] = recipe.carbohydrate_content if recipe.carbohydrate_content is not None else 0.0
    features['FiberContent'] = recipe.fiber_content if recipe.fiber_content is not None else 0.0
    features['SugarContent'] = recipe.sugar_content if recipe.sugar_content is not None else 0.0
    features['ProteinContent'] = recipe.protein_content if recipe.protein_content is not None else 0.0
    
    features['PrepTime'] = recipe.prep_time if recipe.prep_time is not None else 15.0
    features['CookTime'] = recipe.cook_time if recipe.cook_time is not None else 30.0
    features['TotalTime'] = recipe.total_time if recipe.total_time is not None else 45.0
    
    features['RecipeServings'] = recipe.recipe_servings if recipe.recipe_servings is not None else 4
    
    if recipe.recipe_category:
        common_categories = ['Dessert', 'Main Dish', 'Side Dish', 'Appetizer', 
                           'Salad', 'Soup', 'Breakfast', 'Beverage', 'Snack']
        for cat in common_categories:
            features[f'Category_{cat}'] = 1.0 if recipe.recipe_category.lower() == cat.lower() else 0.0
    else:
        common_categories = ['Dessert', 'Main Dish', 'Side Dish', 'Appetizer', 
                           'Salad', 'Soup', 'Breakfast', 'Beverage', 'Snack']
        for cat in common_categories:
            features[f'Category_{cat}'] = 0.0
    
    normalized_ingredients = [ing.lower().strip() for ing in recipe.ingredients]
    
    if ingredient_encoder is not None and hasattr(ingredient_encoder, 'classes_'):
        for ingredient in ingredient_encoder.classes_:
            features[ingredient] = 1.0 if ingredient in normalized_ingredients else 0.0
    else:
        common_ingredients = [
            'salt', 'pepper', 'butter', 'oil', 'flour', 'sugar', 'eggs', 
            'milk', 'cheese', 'onion', 'garlic', 'tomato', 'chicken', 'beef'
        ]
        for ing in common_ingredients:
            features[ing] = 1.0 if ing in normalized_ingredients else 0.0
    
    df = pd.DataFrame([features])
    
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_columns]
    
    return df


def predict_popularity(recipe: Recipe) -> Dict:
    try:
        nutrition_cols = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                          'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
        time_cols = ['PrepTime', 'CookTime', 'TotalTime']
        servings_col = ['RecipeServings']
        
        expected_features = nutrition_cols + time_cols + servings_col
        
        features_dict = {}
        features_dict['Calories'] = recipe.calories if recipe.calories is not None else 0.0
        features_dict['FatContent'] = recipe.fat_content if recipe.fat_content is not None else 0.0
        features_dict['SaturatedFatContent'] = recipe.saturated_fat_content if recipe.saturated_fat_content is not None else 0.0
        features_dict['CholesterolContent'] = recipe.cholesterol_content if recipe.cholesterol_content is not None else 0.0
        features_dict['SodiumContent'] = recipe.sodium_content if recipe.sodium_content is not None else 0.0
        features_dict['CarbohydrateContent'] = recipe.carbohydrate_content if recipe.carbohydrate_content is not None else 0.0
        features_dict['FiberContent'] = recipe.fiber_content if recipe.fiber_content is not None else 0.0
        features_dict['SugarContent'] = recipe.sugar_content if recipe.sugar_content is not None else 0.0
        features_dict['ProteinContent'] = recipe.protein_content if recipe.protein_content is not None else 0.0
        features_dict['PrepTime'] = recipe.prep_time if recipe.prep_time is not None else 15.0
        features_dict['CookTime'] = recipe.cook_time if recipe.cook_time is not None else 30.0
        features_dict['TotalTime'] = recipe.total_time if recipe.total_time is not None else 45.0
        features_dict['RecipeServings'] = recipe.recipe_servings if recipe.recipe_servings is not None else 4
        
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[expected_features]
        
        if scaler is not None and hasattr(scaler, 'mean_'):
            features_scaled = scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        if model is not None and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]
            
            is_popular = bool(prediction)
            popularity_probability = float(proba[1] if len(proba) > 1 else proba[0])
        else:
            is_popular = False
            popularity_probability = 0.5
        
        return {
            'is_popular': is_popular,
            'popularity_probability': popularity_probability,
            'recipe_name': recipe.name,
            'message': 'Recipe is predicted to be popular' if is_popular else 'Recipe is predicted to be not popular'
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'message': 'Error making prediction'
        }


@app.route('/predict', methods=['POST'])
def popular():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'name' not in data or 'ingredients' not in data:
            return jsonify({'error': 'Missing required fields: name and ingredients'}), 400
        
        recipe = Recipe(
            name=data['name'],
            ingredients=data['ingredients'],
            recipe_category=data.get('recipe_category'),
            recipe_servings=data.get('recipe_servings'),
            calories=data.get('calories'),
            fat_content=data.get('fat_content'),
            saturated_fat_content=data.get('saturated_fat_content'),
            cholesterol_content=data.get('cholesterol_content'),
            sodium_content=data.get('sodium_content'),
            carbohydrate_content=data.get('carbohydrate_content'),
            fiber_content=data.get('fiber_content'),
            sugar_content=data.get('sugar_content'),
            protein_content=data.get('protein_content'),
            prep_time=data.get('prep_time'),
            cook_time=data.get('cook_time'),
            total_time=data.get('total_time'),
            description=data.get('description'),
            instructions=data.get('instructions')
        )
        
        result = predict_popularity(recipe)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error processing request'
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Loading Machine Learning Model")
    print("=" * 60)
    load_model()
    print("=" * 60)
    print("Starting Flask server...")
    print("API available at: http://localhost:5000")
    print("Endpoint: POST /popular")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

