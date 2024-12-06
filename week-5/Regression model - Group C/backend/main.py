from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/plot_data")
def get_plot_data():
    iris = load_iris()
    x_data = iris.data

    x_sepal_length = x_data[:, 0].reshape(-1, 1)
    y_sepal_width = x_data[:, 1]
    x_train_sep_len, x_test_sep_len, y_train_sep_width, y_test_sep_width = train_test_split(
        x_sepal_length, y_sepal_width, test_size=0.2, random_state=1
    )

    model_sepal_width = LinearRegression()
    model_sepal_width.fit(x_train_sep_len, y_train_sep_width)
    y_pred_sep_width = model_sepal_width.predict(x_test_sep_len)

    r2_sepal_width = r2_score(y_test_sep_width, y_pred_sep_width)
    mse_sepal_width = mean_squared_error(y_test_sep_width, y_pred_sep_width)

    x_sepal_dims = x_data[:, 0:2]
    y_petal_dims = x_data[:, 2:4]
    x_train_sepal_dims, x_test_sepal_dims, y_train_petal_dims, y_test_petal_dims = train_test_split(
        x_sepal_dims, y_petal_dims, test_size=0.2, random_state=42
    )

    model_petal = LinearRegression()
    model_petal.fit(x_train_sepal_dims, y_train_petal_dims)
    y_pred_petal_dims = model_petal.predict(x_test_sepal_dims)

    r2_petal_length = r2_score(y_test_petal_dims[:, 0], y_pred_petal_dims[:, 0])
    mse_petal_length = mean_squared_error(y_test_petal_dims[:, 0], y_pred_petal_dims[:, 0])

    r2_petal_width = r2_score(y_test_petal_dims[:, 1], y_pred_petal_dims[:, 1])
    mse_petal_width = mean_squared_error(y_test_petal_dims[:, 1], y_pred_petal_dims[:, 1])

    return {
        "model_sepal_width": {
            "r2_score": r2_sepal_width,
            "mse": mse_sepal_width,
            "x_test_sepal_length": x_test_sep_len.tolist(),
            "y_test_sepal_width": y_test_sep_width.tolist(),
            "y_pred_sepal_width": y_pred_sep_width.tolist()
        },
        "model_petal_dims": {
            "r2_score_petal_length": r2_petal_length,
            "mse_petal_length": mse_petal_length,
            "r2_score_petal_width": r2_petal_width,
            "mse_petal_width": mse_petal_width,
            "x_test_sepal_dims": x_test_sepal_dims.tolist(),
            "y_test_petal_dims": y_test_petal_dims.tolist(),
            "y_pred_petal_dims": y_pred_petal_dims.tolist()
        }
    }