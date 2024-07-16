from flask import Flask, render_template_string, jsonify
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)

@app.route('/data')
def data():
    # Generate dummy 4000-dimensional embeddings for 10 sentences
    np.random.seed(42)  # For reproducibility
    num_sentences = 10  # Number of sentences
    dim = 4000  # Dimensionality of embeddings
    sentences = [f"sentence {i}" for i in range(num_sentences)]
    embeddings = np.random.rand(num_sentences, dim)

    # Reduce dimensionality to 3D using PCA
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings).tolist()

    data = {
        "sentences": sentences,
        "vectors": reduced_embeddings
    }
    return jsonify(data)

@app.route('/')
def index():
    # HTML template as a string
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>3D Sentence Embeddings Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>3D Sentence Embeddings Visualization</h1>
        <div id="plot" style="width: 100%; height: 100vh;"></div>
        <script>
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    var sentences = data.sentences;
                    var vectors = data.vectors;

                    var trace_points = {
                        x: vectors.map(point => point[0]),
                        y: vectors.map(point => point[1]),
                        z: vectors.map(point => point[2]),
                        mode: 'markers+text',
                        type: 'scatter3d',
                        text: sentences,
                        textposition: 'top center',
                        marker: {
                            size: 8,
                            opacity: 0.8
                        }
                    };

                    var trace_lines = {
                        x: vectors.flatMap((point, index) => index > 0 ? [vectors[index-1][0], point[0], null] : []),
                        y: vectors.flatMap((point, index) => index > 0 ? [vectors[index-1][1], point[1], null] : []),
                        z: vectors.flatMap((point, index) => index > 0 ? [vectors[index-1][2], point[2], null] : []),
                        mode: 'lines',
                        type: 'scatter3d',
                        line: {
                            color: 'rgba(100, 100, 100, 0.8)',
                            width: 2
                        }
                    };

                    var layout = {
                        title: '3D Sentence Embeddings Visualization',
                        showlegend: false
                    };

                    Plotly.newPlot('plot', [trace_points, trace_lines], layout);
                });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

if __name__ == "__main__":
    app.run(debug=True)
