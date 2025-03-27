import flask
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/0.08-16.12-0.01-0.92.keras', custom_objects={'weighted_mse': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))})

# Load the training data for scaling
df = pd.read_csv('data/processed/train.csv')

# Instantiate the QuantileTransformer
price_transformer = QuantileTransformer(n_quantiles=150)

# Fit the transformer to the 'SalePrice' column in the training data
df['SalePrice'] = price_transformer.fit_transform(df[['SalePrice']])

x = df.drop('SalePrice', axis=1)

# Split the data into training and testing sets (you only need x_train here)
x_train = x

transformer = QuantileTransformer(n_quantiles=150).fit(x_train.select_dtypes(include=['int64', 'float64']))

numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns  # Get numerical columns from training set

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Convert the input data to a NumPy array
        input_data = np.array([
            data['LotFrontage'], data['LotArea'], data['YearBuilt'], data['YearRemodAdd'],
            data['MasVnrArea'], data['BsmtFinSF1'], data['BsmtFinSF2'], data['BsmtUnfSF'],
            data['TotalBsmtSF'], data['1stFlrSF'], data['2ndFlrSF'], data['LowQualFinSF'],
            data['GrLivArea'], data['GarageYrBlt'], data['GarageArea'], data['WoodDeckSF'],
            data['OpenPorchSF'], data['EnclosedPorch'], data['3SsnPorch'], data['ScreenPorch'],
            data['MiscVal'], data['PoolQC_0'], data['PoolQC_1'], data['PoolQC_2'], data['PoolQC_3'],
            data['MiscFeature_0'], data['MiscFeature_1'], data['MiscFeature_2'], data['MiscFeature_3'],
            data['MiscFeature_4'], data['Alley_0'], data['Alley_1'], data['Alley_2'], data['Fence_0'],
            data['Fence_1'], data['Fence_2'], data['Fence_3'], data['Fence_4'], data['MasVnrType_0'],
            data['MasVnrType_1'], data['MasVnrType_2'], data['MasVnrType_3'], data['FireplaceQu_0'],
            data['FireplaceQu_1'], data['FireplaceQu_2'], data['FireplaceQu_3'], data['FireplaceQu_4'],
            data['FireplaceQu_5'], data['GarageQual_0'], data['GarageQual_1'], data['GarageQual_2'],
            data['GarageQual_3'], data['GarageQual_4'], data['GarageQual_5'], data['GarageFinish_0'],
            data['GarageFinish_1'], data['GarageFinish_2'], data['GarageFinish_3'], data['GarageType_0'],
            data['GarageType_1'], data['GarageType_2'], data['GarageType_3'], data['GarageType_4'],
            data['GarageType_5'], data['GarageType_6'], data['GarageCond_0'], data['GarageCond_1'],
            data['GarageCond_2'], data['GarageCond_3'], data['GarageCond_4'], data['GarageCond_5'],
            data['BsmtFinType2_0'], data['BsmtFinType2_1'], data['BsmtFinType2_2'], data['BsmtFinType2_3'],
            data['BsmtFinType2_4'], data['BsmtFinType2_5'], data['BsmtFinType2_6'], data['BsmtExposure_0'],
            data['BsmtExposure_1'], data['BsmtExposure_2'], data['BsmtExposure_3'], data['BsmtExposure_4'],
            data['BsmtCond_0'], data['BsmtCond_1'], data['BsmtCond_2'], data['BsmtCond_3'], data['BsmtCond_4'],
            data['BsmtQual_0'], data['BsmtQual_1'], data['BsmtQual_2'], data['BsmtQual_3'], data['BsmtQual_4'],
            data['BsmtFinType1_0'], data['BsmtFinType1_1'], data['BsmtFinType1_2'], data['BsmtFinType1_3'],
            data['BsmtFinType1_4'], data['BsmtFinType1_5'], data['BsmtFinType1_6'], data['Electrical_0'],
            data['Electrical_1'], data['Electrical_2'], data['Electrical_3'], data['Electrical_4'],
            data['Condition2_0'], data['Condition2_1'], data['Condition2_2'], data['Condition2_3'],
            data['Condition2_4'], data['Condition2_5'], data['Condition2_6'], data['Condition2_7'],
            data['BldgType_0'], data['BldgType_1'], data['BldgType_2'], data['BldgType_3'], data['BldgType_4'],
            data['Neighborhood_0'], data['Neighborhood_1'], data['Neighborhood_2'], data['Neighborhood_3'],
            data['Neighborhood_4'], data['Neighborhood_5'], data['Neighborhood_6'], data['Neighborhood_7'],
            data['Neighborhood_8'], data['Neighborhood_9'], data['Neighborhood_10'], data['Neighborhood_11'],
            data['Neighborhood_12'], data['Neighborhood_13'], data['Neighborhood_14'], data['Neighborhood_15'],
            data['Neighborhood_16'], data['Neighborhood_17'], data['Neighborhood_18'], data['Neighborhood_19'],
            data['Neighborhood_20'], data['Neighborhood_21'], data['Neighborhood_22'], data['Neighborhood_23'],
            data['Neighborhood_24'], data['LandSlope_0'], data['LandSlope_1'], data['LandSlope_2'],
            data['LotConfig_0'], data['LotConfig_1'], data['LotConfig_2'], data['LotConfig_3'], data['LotConfig_4'],
            data['Condition1_0'], data['Condition1_1'], data['Condition1_2'], data['Condition1_3'],
            data['Condition1_4'], data['Condition1_5'], data['Condition1_6'], data['Condition1_7'],
            data['Condition1_8'], data['LandContour_0'], data['LandContour_1'], data['LandContour_2'],
            data['LandContour_3'], data['LotShape_0'], data['LotShape_1'], data['LotShape_2'], data['LotShape_3'],
            data['Street_0'], data['Street_1'], data['MSZoning_0'], data['MSZoning_1'], data['MSZoning_2'],
            data['MSZoning_3'], data['MSZoning_4'], data['Utilities_0'], data['Utilities_1'], data['HouseStyle_0'],
            data['HouseStyle_1'], data['HouseStyle_2'], data['HouseStyle_3'], data['HouseStyle_4'],
            data['HouseStyle_5'], data['HouseStyle_6'], data['HouseStyle_7'], data['Foundation_0'],
            data['Foundation_1'], data['Foundation_2'], data['Foundation_3'], data['Foundation_4'],
            data['Foundation_5'], data['ExterQual_0'], data['ExterQual_1'], data['ExterQual_2'], data['ExterQual_3'],
            data['ExterCond_0'], data['ExterCond_1'], data['ExterCond_2'], data['ExterCond_3'], data['ExterCond_4'],
            data['Heating_0'], data['Heating_1'], data['Heating_2'], data['Heating_3'], data['Heating_4'],
            data['Heating_5'], data['KitchenQual_0'], data['KitchenQual_1'], data['KitchenQual_2'],
            data['KitchenQual_3'], data['Functional_0'], data['Functional_1'], data['Functional_2'],
            data['Functional_3'], data['Functional_4'], data['Functional_5'], data['Functional_6'],
            data['PavedDrive_0'], data['PavedDrive_1'], data['PavedDrive_2'], data['SaleType_0'],
            data['SaleType_1'], data['SaleType_2'], data['SaleType_3'], data['SaleType_4'], data['SaleType_5'],
            data['SaleType_6'], data['SaleType_7'], data['SaleType_8'], data['SaleCondition_0'],
            data['SaleCondition_1'], data['SaleCondition_2'], data['SaleCondition_3'], data['SaleCondition_4'],
            data['SaleCondition_5'], data['Exterior1st_0'], data['Exterior1st_1'], data['Exterior1st_2'],
            data['Exterior1st_3'], data['Exterior1st_4'], data['Exterior1st_5'], data['Exterior1st_6'],
            data['Exterior1st_7'], data['Exterior1st_8'], data['Exterior1st_9'], data['Exterior1st_10'],
            data['Exterior1st_11'], data['Exterior1st_12'], data['Exterior1st_13'], data['Exterior1st_14'],
            data['Exterior2nd_0'], data['Exterior2nd_1'], data['Exterior2nd_2'], data['Exterior2nd_3'],
            data['Exterior2nd_4'], data['Exterior2nd_5'], data['Exterior2nd_6'], data['Exterior2nd_7'],
            data['Exterior2nd_8'], data['Exterior2nd_9'], data['Exterior2nd_10'], data['Exterior2nd_11'],
            data['Exterior2nd_12'], data['Exterior2nd_13'], data['Exterior2nd_14'], data['Exterior2nd_15'],
            data['RoofStyle_0'], data['RoofStyle_1'], data['RoofStyle_2'], data['RoofStyle_3'], data['RoofStyle_4'],
            data['RoofStyle_5'], data['RoofMatl_0'], data['RoofMatl_1'], data['RoofMatl_2'], data['RoofMatl_3'],
            data['RoofMatl_4'], data['RoofMatl_5'], data['RoofMatl_6'], data['RoofMatl_7'], data['CentralAir_0'],
            data['CentralAir_1'], data['HeatingQC_0'], data['HeatingQC_1'], data['HeatingQC_2'],
            data['HeatingQC_3'], data['HeatingQC_4'], data['MSSubClass_0'], data['MSSubClass_1'],
            data['MSSubClass_2'], data['MSSubClass_3'], data['MSSubClass_4'], data['MSSubClass_5'],
            data['MSSubClass_6'], data['MSSubClass_7'], data['MSSubClass_8'], data['MSSubClass_9'],
            data['MSSubClass_10'], data['MSSubClass_11'], data['MSSubClass_12'], data['MSSubClass_13'],
            data['MSSubClass_14'], data['OverallCond_0'], data['OverallCond_1'], data['OverallCond_2'],
            data['OverallCond_3'], data['OverallCond_4'], data['OverallCond_5'], data['OverallCond_6'],
            data['OverallCond_7'], data['OverallCond_8'], data['OverallQual_0'], data['OverallQual_1'],
            data['OverallQual_2'], data['OverallQual_3'], data['OverallQual_4'], data['OverallQual_5'],
            data['OverallQual_6'], data['OverallQual_7'], data['OverallQual_8'], data['OverallQual_9'],
            data['MoSold_0'], data['MoSold_1'], data['MoSold_2'], data['MoSold_3'], data['MoSold_4'],
            data['MoSold_5'], data['MoSold_6'], data['MoSold_7'], data['MoSold_8'], data['MoSold_9'],
            data['MoSold_10'], data['MoSold_11'], data['YrSold_0'], data['YrSold_1'], data['YrSold_2'],
            data['YrSold_3'], data['YrSold_4'], data['KitchenAbvGr_0'], data['KitchenAbvGr_1'],
            data['KitchenAbvGr_2'], data['KitchenAbvGr_3'], data['FullBath_0'], data['FullBath_1'],
            data['FullBath_2'], data['FullBath_3'], data['HalfBath_0'], data['HalfBath_1'], data['HalfBath_2'],
            data['BsmtFullBath_0'], data['BsmtFullBath_1'], data['BsmtFullBath_2'], data['BsmtFullBath_3'],
            data['BsmtHalfBath_0'], data['BsmtHalfBath_1'], data['BsmtHalfBath_2'], data['BedroomAbvGr_0'],
            data['BedroomAbvGr_1'], data['BedroomAbvGr_2'], data['BedroomAbvGr_3'], data['BedroomAbvGr_4'],
            data['BedroomAbvGr_5'], data['BedroomAbvGr_6'], data['BedroomAbvGr_7'], data['Fireplaces_0'],
            data['Fireplaces_1'], data['Fireplaces_2'], data['Fireplaces_3'], data['GarageCars_0'],
            data['GarageCars_1'], data['GarageCars_2'], data['GarageCars_3'], data['GarageCars_4'],
            data['TotRmsAbvGrd_0'], data['TotRmsAbvGrd_1'], data['TotRmsAbvGrd_2'], data['TotRmsAbvGrd_3'],
            data['TotRmsAbvGrd_4'], data['TotRmsAbvGrd_5'], data['TotRmsAbvGrd_6'], data['TotRmsAbvGrd_7'],
            data['TotRmsAbvGrd_8'], data['TotRmsAbvGrd_9'], data['TotRmsAbvGrd_10'], data['TotRmsAbvGrd_11'],
            data['PoolArea_0'], data['PoolArea_1'], data['PoolArea_2'], data['PoolArea_3'], data['PoolArea_4'],
            data['PoolArea_5'], data['PoolArea_6'], data['PoolArea_7']
        ]).reshape(1, -1)

        # Transform the input data using the pre-fitted QuantileTransformer
        input_df = pd.DataFrame(input_data, columns=x_train.columns)  # Create DataFrame for transformation
        input_df[numerical_cols] = transformer.transform(input_df[numerical_cols])
        input_data = input_df.values  # Convert back to numpy array

        # Make the prediction
        prediction = model.predict(input_data)

        # Inverse transform the prediction
        predicted_price = price_transformer.inverse_transform(prediction)[0][0]

        # Return the predicted sale price as a JSON response
        return jsonify({'predicted_sale_price': float(predicted_price)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)