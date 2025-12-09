from keras_image_helper import create_preprocessor
import numpy as np
import onnxruntime as ort


def preprocess_pytorch_style(X):
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW (batch, height, width, channels → batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)

def predict(image_url: str):
    preprocessor = create_preprocessor(preprocess_pytorch_style, target_size=(200, 200))
    X = preprocessor.from_url(image_url)


    session = ort.InferenceSession("./model/hair_classifier_v1.onnx")

    input_name = session.get_inputs()[0].name

    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: X})[0]

    classes = ["straight", "curly"]

    float_predictions = pred[0].tolist()

    result = dict(zip(classes, float_predictions))

    return {
            "statusCode": 200,
            "body": result
    }

if __name__ == "__main__":
    image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    print(predict(image_url))