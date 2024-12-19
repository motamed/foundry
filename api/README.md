# Foundry API

This is a Flask-based API for managing and retrieving model information.</br>
The API supports fetching models from a URL or a local file, and provides several endpoints for interacting with the models.

## Setup

1. Clone the repository.
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Create a `.env` file in the root directory and add the following variables:
    ```properties
    MODELS_FILE_URL=<URL_to_JSON_file>
    MODELS_FILE_PATH=./../json/full.json
    ```

## Endpoints

### Redirect to Swagger UI

- **URL:** `/`
- **Method:** `GET`
- **Description:** Redirects to the Swagger UI documentation.

### Get All Models

- **URL:** `/models`
- **Method:** `GET`
- **Description:** Retrieves all models from the specified URL or local file.
- **Response:**
    - `200 OK` with a JSON array of models.
    - `404 Not Found` if unable to fetch models.

### Get Model by ID or Name

- **URL:** `/model`
- **Method:** `GET`
- **Description:** Retrieves a model by its **ID** or **Name**.
- **Query Parameters:**
    - `id` : The ID of the model.
    - `name` : The name of the model.
- **Response:**
    - `200 OK` with the model data.
    - `400 Bad Request` if neither `id` nor `name` is provided.
    - `404 Not Found` if the model is not found.

### Count Models

- **URL:** `/count`
- **Method:** `GET`
- **Description:** Returns the count of models.
- **Response:**
    - `200 OK` with the count of models.
    - `404 Not Found` if unable to fetch models.

### Search Models by Name

- **URL:** `/search`
- **Method:** `GET`
- **Description:** Searches for models by name.
- **Query Parameters:**
    - `name` (required): The search term for the model name.
- **Response:**
    - `200 OK` with a JSON array of matching models.
    - `404 Not Found` if unable to fetch models.

## Swagger UI

- **URL:** `/swagger`
- **Description:** Provides an interactive Swagger UI for testing the API endpoints.
- **Configuration:**
    - The OpenAPI file is located at `/static/swagger.yaml`.

## Running the Application

To run the application, execute the following command:

```sh
python app.py
```

The application will be available at `http://127.0.0.1:5000/`.
