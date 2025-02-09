import uvicorn
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def deploy_model(data, *args, **kwargs):
    """
    Deploy trained model using FastAPI.

    Args:
        data (dict): Contains trained model & scaler.
    
    Returns:
        str: Deployment confirmation message.
    """

    # Retrieve trained model & scaler
    model = data.get('model')
    scaler = data.get('scaler')


    if model is None or scaler is None:
        raise ValueError("Model or Scaler is missing!")

    # Initialize FastAPI app
    app = FastAPI()


    @app.post("/predict/")
    def predict(Clump_Thickness : int, Uniformity_Cell_Size : int, Uniformity_Cell_Shape : int,
    Marginal_Adhesion : int, Single_Epithelial_Cell_Size : int, Bare_Nuclei : int,
    Bland_Chromatin : int, Normal_Nucleoli : int, Mitoses : int):
        """
        Predict whether a tumor is benign or malignant.
        """
        try:
            # Convert input to DataFrame
            input_data = pd.DataFrame([[Clump_Thickness, Uniformity_Cell_Size, Uniformity_Cell_Shape, 
                                    Marginal_Adhesion, Single_Epithelial_Cell_Size, Bare_Nuclei, 
                                    Bland_Chromatin, Normal_Nucleoli, Mitoses]],
                                  columns=['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                           'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                                           'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'])


            input_data_scaled = scaler.transform(input_data)

            print(input_data_scaled.shape)

            # Make prediction
            prediction = model.predict(input_data_scaled)
            result = "Malignant" if prediction[0] == 1 else "Benign"
            
            return {"prediction": result}

        except Exception as e:
            return {"error": str(e)}

    # Run FastAPI in a separate thread
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    run_api()

    # thread = threading.Thread(target=run_api, daemon=True)
    # thread.start()

    return "Model deployed at http://0.0.0.0:8000 🚀"


@test
def test_output(output, *args) -> None:
    """
    Test if deployment message is returned.
    """
    assert output is not None, 'Deployment did not start'
