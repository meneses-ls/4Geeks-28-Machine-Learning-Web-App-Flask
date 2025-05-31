from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(MODEL_PATH)

INPUT_RANGES = {
    'val1': (0, 20),
    'val2': (0, 200),
    'val3': (0, 150),
    'val4': (0, 100),
    'val5': (0, 900),
    'val6': (0, 70),
    'val7': (0, 2),
    'val8': (0, 120)
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            inputs = []
            for i in range(1, 9):
                key = f'val{i}'
                value = float(request.form[key])
                min_val, max_val = INPUT_RANGES[key]

                if not (min_val <= value <= max_val):
                    return render_template('index.html', prediction_text=f'Error: {key} debe estar entre {min_val} y {max_val}.')
                
                inputs.append(value)

            input_data = np.array([inputs])
            prediction = model.predict(input_data)
            return render_template('index.html', prediction_text=f'Resultado: {prediction[0]}')

        except ValueError:
            return render_template('index.html', prediction_text='Error: Ingresa solo valores numéricos válidos.')

    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)