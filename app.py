import dash
from dash import dcc, html, Input, Output, State
import threading
import webview

import joblib
log_model = joblib.load("outputs/logistic_model.pkl")
NB_model = joblib.load("outputs/nb_model.pkl")
vectorizer = joblib.load("outputs/tfidf_vectorizer.pkl")
# =========================
# Create Dash App
# =========================
app = dash.Dash(__name__, assets_folder="assets")



app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Sentiment App</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                width: 100%;
                height: 100%;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    style={
        "backgroundImage": "url('/assets/NLP3.jpg')",
        "backgroundSize": "cover",
        "width": "100vw",
        "height": "100vh",
        "padding": "0px",
        "margin": "0px",
        "overflow": "hidden"
    },

    children=[

        html.H1(
            "Sentiment Analysis Desktop App",
            style={
                "textAlign": "center",
                "color": "#8cc9a1",
                "margin-top": "10px"
            }
        ),

        dcc.Textarea(
            id="input_text",
            placeholder="Enter your review here...",
            style={
                "width": "60%",
                "height": "160px",
                "display": "block",
                "margin": "auto",
                "padding": "15px",
                "fontSize": "18px",
                "borderRadius": "15px",
                "border_color": "cyan",
                "margin-top": "10px",
                "backgroundColor": "rgb(91 102 84 / 70%)",
                "color": "#98d6ab"



            }
        ),

        html.Br(),

        html.Div([

            html.Button(
                "Predict Logistic",
                id="btn_log",
                n_clicks=0,
                style={
                    "backgroundColor": "#153b36",
                    "color": "#8cc9a1",
                    "padding": "12px 25px",
                    "fontSize": "18px",
                    "borderRadius": "20px",
                    "margin": "20px",
                    "cursor": "pointer"

                }
            ),

            html.Button(
                "Predict Naive Bayes",
                id="btn_nb",
                n_clicks=0,
                style={
                    "backgroundColor": "#302610",
                    "color": "#8cc9a1",
                    "padding": "12px 25px",
                    "fontSize": "18px",
                    "borderRadius": "20px",
                    "margin": "20px",
                    "cursor": "pointer"

                }
            )

        ], style={"textAlign": "center"}),

        html.Br(),

        html.Div(
            id="output_box",
            style={"display": "none"}
        )
    ]
)

# =========================
# Prediction
# =========================
@app.callback(
    Output("output_box", "children"),
    Output("output_box", "style"),

    Input("btn_log", "n_clicks"),
    Input("btn_nb", "n_clicks"),

    State("input_text", "value")
)

def predict(btn_log, btn_nb, text):

    if not text:
        return "", {"display": "none"}

    ctx = dash.callback_context

    if not ctx.triggered:
        return "", {"display": "none"}

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    vec = vectorizer.transform([text])

    if button_id == "btn_log":
        pred = log_model.predict(vec)[0]
        model_name = "Logistic Regression"
    else:
        pred = NB_model.predict(vec)[0]
        model_name = "Naive Bayes"

    if pred == 1:
        result = "Positive 😊"
        color = "green"
    else:
        result = "Negative 😡"
        color = "red"

    return (
        f"{model_name}: {result}",
        {
            "width": "50%",
            "margin": "auto",
            "padding": "20px",
            "textAlign": "center",
            "fontSize": "24px",
            "backgroundColor": "black",
            "borderRadius": "15px",
            "color": color,
            "display": "block",
            "fontWeight": "bold"
        }
    )

# =========================
# Run Dash Server
# =========================
def run_dash():
    app.run(debug=False)

# =========================
# Main Desktop Window
# =========================
if __name__ == '__main__':

    t = threading.Thread(target=run_dash)
    t.daemon = True
    t.start()

    webview.create_window(
        "Sentiment Analysis App",
        "http://127.0.0.1:8050/",
        width=1200,
        height=800
    )

    webview.start()