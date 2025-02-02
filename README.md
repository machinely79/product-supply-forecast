# Product Delivery Forecasting in Supply Chain based on LightGBM Regression

This application implements a **LightGBM regression model** for **multivariate and multi-output forecasting** in supply chain management. It predicts the next **delivery quantity and its expected date at a hierarchical level**, covering various products and delivery locations. The model predicts the upcoming deliveries and delivery dates for approximately 5,000 locations and around 800 products.

![SCMap](https://github.com/machinely79/product-supply-forecast/blob/main/images/SCMap.png)

One of its key advantages is handling **intermittent time series**, where delivery event data is sparse and non-continuous. By leveraging a **hybrid machine learning approach**, the system enhances **inventory management** and **logistics efficiency**, helping businesses make more accurate and data-driven supply chain decisions.

## Business Problem

Daily deliveries of items lead to returns and write-offs of certain quantities due to inaccurate demand forecasting by sales personnel. Around **13% returns** (on average) on the total number of delivered items.  

`[(Total delivered – Total sold) / Total delivered] * 100`  

On average, about **800 more items** were delivered daily than needed for sales.

![Deliveries](https://github.com/machinely79/product-supply-forecast/blob/main/images/Deliveries.png)


## Data, Model Training and Testing

#### Training and Validation: The model was trained and validated using data from **January 1, 2018 to December 30, 2020**. A **five-fold cross-validation** approach was applied. The average daily error (returns) per split ranged between **300 and 600 items**.

#### Testing: The model was tested on data from **January 1, 2021 to December 31, 2021**. The observed average daily error (returns) was approximately **400 items**.

#### Note:  
Only a subset of the dataset has been uploaded to the repository for demonstration purposes. The full dataset is not included due to size and confidentiality constraints.

## Achieving Business Value: 7% Reduction in Returns  
The application has **reduced returns (write-off costs) by an average of approximately 7%**.
