import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import torch
import gradio as gr
import html
import numpy as np

# Load translation model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-is-en")

# Load models and tokenizers
model_name_toxic = "s-nlp/roberta_toxicity_classifier"
tokenizer_toxic = transformers.AutoTokenizer.from_pretrained(model_name_toxic)
model_toxic = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_toxic)
model_toxic.eval()

model_name_sentiment = "Birkir/electra-base-igc-is-sentiment-analysis"
tokenizer_sentiment = transformers.AutoTokenizer.from_pretrained(model_name_sentiment)
model_sentiment = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_sentiment)
model_sentiment.eval()

model_name_emotion = "SamLowe/roberta-base-go_emotions"
tokenizer_emotion = transformers.AutoTokenizer.from_pretrained(model_name_emotion)
model_emotion = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_emotion)
model_emotion.eval()

model_name_formality = "s-nlp/roberta-base-formality-ranker"
tokenizer_formality = transformers.AutoTokenizer.from_pretrained(model_name_formality)
model_formality = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_formality)
model_formality.eval()

# Pipelines
pipeline_toxic = transformers.pipeline(
    "text-classification",
    model=model_toxic,
    tokenizer=tokenizer_toxic,
    device=0,
    top_k=None
)

pipeline_sentiment = transformers.pipeline(
    "text-classification",
    model=model_sentiment,
    tokenizer=tokenizer_sentiment,
    device=0,
    top_k=None
)

pipeline_emotion = transformers.pipeline(
    "text-classification",
    model=model_emotion,
    tokenizer=tokenizer_emotion,
    device=0,
    top_k=None
)

pipeline_formality = transformers.pipeline(
    "text-classification",
    model=model_formality,
    tokenizer=tokenizer_formality,
    device=0,
    top_k=None
)

# SHAP Explainers 
explainer_toxic = shap.Explainer(pipeline_toxic, tokenizer_toxic)
explainer_sentiment = shap.Explainer(pipeline_sentiment, tokenizer_sentiment)
explainer_formality = shap.Explainer(pipeline_formality, tokenizer_formality)

# Labels for each classifier 
class_labels_formality = ["formal", "informal"]
class_labels_sentiment = ["Negative", "Positive"]
class_labels_toxic = ["neutral", "toxic"]
class_labels_emotion = [
    "disappointment", "sadness", "annoyance", "neutral", "disapproval",
    "realization", "nervousness", "approval", "joy", "anger", "embarrassment",
    "caring", "remorse", "disgust", "grief", "confusion", "relief", "desire",
    "admiration", "optimism", "fear", "love", "excitement", "curiosity",
    "amusement", "surprise", "gratitude", "pride"
]

# Icelandic translation mapping
label_map = {
    "disappointment": "Vonbrigði",
    "sadness": "Depurð",
    "annoyance": "Pirringur",
    "neutral": "Hlutlaus",
    "disapproval": "Vanþóknun",
    "realization": "Uppgötvun",
    "nervousness": "Stress",
    "approval": "Samþykki",
    "joy": "Gleði",
    "anger": "Reiði",
    "embarrassment": "Vandræðaleiki",
    "caring": "Umhyggjusemi",
    "remorse": "Iðrun",
    "disgust": "Andstyggð",
    "grief": "Sorg",
    "confusion": "Ruglingur",
    "relief": "Léttir",
    "desire": "Löngun",
    "admiration": "Aðdáun",
    "optimism": "Bjartsýni",
    "fear": "Ótti",
    "love": "Ást",
    "excitement": "Spenna",
    "curiosity": "Forvitni",
    "amusement": "Skemmtun",
    "surprise": "Furða",
    "gratitude": "Þakklæti",
    "pride": "Stolt"
}

# TOXICITY 

def classify_toxicity(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    results = pipeline_toxic(translated_text)[0]
    
    toxic_score = next((item['score'] for item in results if item['label'] == 'toxic'), 0.0)
    neutral_score = 1.0 - toxic_score
    
    toxic_score = round(toxic_score, 2)
    neutral_score = round(neutral_score, 2)
    
    return toxic_score

def explain_toxicity(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    shap_values = explainer_toxic([translated_text], fixed_context=1)

    toxic_index = class_labels_toxic.index("toxic")
    toxic_score = classify_toxicity(sentence)

    if toxic_score == 0.0:
        return "Engin óviðeigandi orð fundust í textanum!"

    words_with_scores = []

    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, toxic_index]):
        escaped_word = html.escape(word)
        
        if shap_value >= 0.8:
            font_size = "xx-large"
            color = "#ff0000"  # Red
        elif shap_value >= 0.4:
            font_size = "x-large"
            color = "#ff3333"  # Light red
        elif shap_value >= 0.1:
            font_size = "large"
            color = "#ff6666"  # Lighter red
        elif shap_value >= 0.03:
            font_size = "medium"
            color = "#ff9999"  # Lightest red
        else:
            font_size = "normal"
            color = "white"  

        weight = "bold" if shap_value >= 0.3 else "normal"
        words_with_scores.append(f"<span style='font-size:{font_size}; font-weight:{weight}; color:{color};'>{escaped_word}</span>")

    explanation_output = ''.join(words_with_scores)
    return explanation_output


# SENTIMENT
def classify_sentiment(sentence):
    results = pipeline_sentiment(sentence)[0]

    sentiment_result = results[0] if results[0]['label'] == 'LABEL_0' else results[1]
    sentiment_score = sentiment_result['score']
    negative_score = sentiment_score if sentiment_result['label'] == 'LABEL_0' else 1.0 - sentiment_score
    positive_score = 1.0 - negative_score
    
    negative_score = round(negative_score, 2)
    positive_score = round(positive_score, 2)
    
    return negative_score, positive_score

def explain_sentiment(sentence):
    negative_score, positive_score = classify_sentiment(sentence)

    shap_values = explainer_sentiment([sentence], fixed_context=1)
    positive_index = class_labels_sentiment.index("Positive")

    words = sentence.split()
    word_shap_values = {word: [] for word in words}

    for token, shap_value in zip(shap_values[0].data, shap_values[0].values[:, positive_index]):
        for word in words:
            if token in word:
                word_shap_values[word].append(shap_value)
                break

    word_max_shap = {word: max(shap_values, key=abs) if shap_values else 0 for word, shap_values in word_shap_values.items()}
    sorted_words_by_shap = sorted(word_max_shap.items(), key=lambda x: x[1], reverse=True)

    if positive_score >= 0.95:
        top_positive_words = [word for word, value in sorted_words_by_shap if value > 0][:3]
        top_negative_words = []
    elif negative_score >= 0.95:
        top_positive_words = []
        top_negative_words = [word for word, value in sorted_words_by_shap if value < 0][:3]
    else:
        top_positive_words = [word for word, value in sorted_words_by_shap if value > 0][:3]
        top_negative_words = [word for word, value in sorted_words_by_shap if value < 0][:3]

    words_with_formatted_scores = []

    for word in words:
        max_shap_value = word_max_shap[word]

        if word in top_positive_words:
            color = "#FFB347"  # more intense orange
            font_size = "x-large"
        elif word in top_negative_words:
            color = "#1E90FF"  # more intense blue 
            font_size = "x-large"
        elif max_shap_value > 0:
            color = "#FFDAB9"  # less intense orange 
            font_size = "large"
        elif max_shap_value < 0:
            color = "#87CEEB"  # less intense blue 
            font_size = "large"
        else:
            color = "#F0F0F0"  # Neutral 
            font_size = "normal"

        weight = "bold" if abs(max_shap_value) >= 0.1 else "normal"

        words_with_formatted_scores.append(
            f"<span style='font-size:{font_size}; font-weight:{weight}; color:{color};'>{html.escape(word)}</span>"
        )

    return ' '.join(words_with_formatted_scores)


# EMOTION
def classify_emotion(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    results = pipeline_emotion(translated_text, top_k=len(class_labels_emotion))
    
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotions = sorted_results[:3]
    
    labels = [label_map[e['label']] for e in top_emotions]
    scores = [round(e['score'], 2) for e in top_emotions]
    
    return (
        gr.update(value=scores[0], label=labels[0]), 
        gr.update(value=scores[1], label=labels[1]), 
        gr.update(value=scores[2], label=labels[2])
    )


# FORMALITY
def classify_formality(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    results = pipeline_formality(translated_text)[0]

    formal_score = next((item['score'] for item in results if item['label'] == 'formal'), 0.0)
    informal_score = 1.0 - formal_score
    
    formal_score = round(formal_score, 2)
    informal_score = round(informal_score, 2)
    
    return formal_score, informal_score

def explain_formality(sentence):
    formal_score, informal_score = classify_formality(sentence)
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    shap_values = explainer_formality([translated_text], fixed_context=1)

    formal_index = class_labels_formality.index("formal")

    words = translated_text.split()
    word_shap_values = {word: [] for word in words}

    for token, shap_value in zip(shap_values[0].data, shap_values[0].values[:, formal_index]):
        for word in words:
            if token in word:
                word_shap_values[word].append(shap_value)
                break

    word_max_shap = {word: max(shap_values, key=abs) if shap_values else 0 for word, shap_values in word_shap_values.items()}
    sorted_words_by_shap = sorted(word_max_shap.items(), key=lambda x: x[1], reverse=True)

    top_formal_words = []
    top_informal_words = []

    if formal_score >= 0.95:
        top_formal_words = [word for word, value in sorted_words_by_shap if value > 0][:3]
    elif informal_score >= 0.95:
        top_informal_words = [word for word, value in sorted_words_by_shap if value < 0][:3]
    else:
        top_formal_words = [word for word, value in sorted_words_by_shap if value > 0][:3]
        top_informal_words = [word for word, value in sorted_words_by_shap if value < 0][:3]

    words_with_formatted_scores = []

    for word in words:
        max_shap_value = word_max_shap[word]

        if word in top_formal_words:
            color = "#FFB347"  # more intense orange
            font_size = "x-large"
        elif word in top_informal_words:
            color = "#1E90FF"  # more intense blue
            font_size = "x-large"
        elif max_shap_value > 0:
            color = "#FFDAB9"  # less intense orange 
            font_size = "large"
        elif max_shap_value < 0:
            color = "#87CEEB"  # less intense blue 
            font_size = "large"
        else:
            color = "#F0F0F0"  # Neutral
            font_size = "normal"

        weight = "bold" if abs(max_shap_value) >= 0.1 else "normal"

        words_with_formatted_scores.append(
            f"<span style='font-size:{font_size}; font-weight:{weight}; color:{color};'>{html.escape(word)}</span>"
        )

    return ' '.join(words_with_formatted_scores)


def classify_all(sentence):
    toxicity = classify_toxicity(sentence)
    sentiment = classify_sentiment(sentence)
    emotion = classify_emotion(sentence)
    formality = classify_formality(sentence)
    return toxicity, sentiment, emotion, formality

def explain_all(sentence):
    toxicity = explain_toxicity(sentence)
    sentiment = explain_sentiment(sentence)
    emotion = explain_emotion(sentence)
    formality = explain_formality(sentence)
    return toxicity, sentiment, emotion, formality

def run_all(sentence):
    toxicity = classify_toxicity(sentence)
    neg_score, pos_score = classify_sentiment(sentence)
    emotion1, emotion2, emotion3 = classify_emotion(sentence)
    formal_score, informal_score = classify_formality(sentence)
    
    toxicity_exp = explain_toxicity(sentence)
    sentiment_exp = explain_sentiment(sentence)
    formality_exp = explain_formality(sentence)
    
    return (
        toxicity, neg_score, pos_score, emotion1, emotion2, emotion3,
        formal_score, informal_score, toxicity_exp, sentiment_exp, formality_exp
    )


# Gradio interface
with gr.Blocks() as iface:
    with gr.Row():
        gr.HTML(
            """
            <center>
                <img src='https://s3.vist.is:10443/hr-cdn/2021/12/HR_logo_vinstri_transparent-1024x298.png' width='250' height='auto'>
            </center>
            """
        )
    with gr.Column():
        input_text = gr.Textbox(lines=3, label='Textinn þinn hér:')
        with gr.Row():
            run_all_btn = gr.Button(value='Skoða allt')
    with gr.Row():
        with gr.Column():
            classify_sentiment_button = gr.Button("Greina lyndi textans")
            classify_sentiment_output_negative = gr.Slider(minimum=0, maximum=1, label="Neikvætt", interactive=False)
            classify_sentiment_output_positive = gr.Slider(minimum=0, maximum=1, label="Jákvætt", interactive=False)
            classify_sentiment_button.click(fn=classify_sentiment, inputs=input_text, outputs=[classify_sentiment_output_negative, classify_sentiment_output_positive])
        with gr.Column():
            explain_sentiment_button = gr.Button("Nánari greining")
            explain_sentiment_output = gr.HTML()
            explain_sentiment_button.click(fn=explain_sentiment, inputs=input_text, outputs=explain_sentiment_output)
    '''with gr.Row():
        with gr.Column():
            classify_emotion_button = gr.Button("Greina algengustu tilfinningarnar í textanum")
            emotion_score1 = gr.Slider(minimum=0, maximum=1, label="", interactive=False)
            emotion_score2 = gr.Slider(minimum=0, maximum=1, label="", interactive=False)
            emotion_score3 = gr.Slider(minimum=0, maximum=1, label="", interactive=False)
            classify_emotion_button.click(fn=classify_emotion, inputs=input_text, outputs=[emotion_score1, emotion_score2, emotion_score3])
        with gr.Column():
            explain_emotion_button = gr.Button("Nánari greining")
            explain_emotion_output = gr.HTML()
            explain_emotion_button.click(fn=explain_emotion, inputs=input_text, outputs=explain_emotion_output)'''
    with gr.Row():
        with gr.Column():
            classify_formality_button = gr.Button("Greina formlegheit textans")
            classify_formality_output_formal = gr.Slider(minimum=0, maximum=1, label="Formlegt", interactive=False)
            classify_formality_output_informal = gr.Slider(minimum=0, maximum=1, label="Óformlegt", interactive=False)
            classify_formality_button.click(fn=classify_formality, inputs=input_text, outputs=[classify_formality_output_formal, classify_formality_output_informal])
        with gr.Column():
            explain_formality_button = gr.Button("Nánari greining")
            explain_formality_output = gr.HTML()
            explain_formality_button.click(fn=explain_formality, inputs=input_text, outputs=explain_formality_output)
    with gr.Row():
        with gr.Column():
            classify_toxicity_button = gr.Button("Greina notkun á óviðeigandi orðum í textanum")
            classify_toxicity_output_toxic = gr.Slider(minimum=0, maximum=1, label="Notkun á óviðeigandi orðum", interactive=False)
            classify_toxicity_button.click(fn=classify_toxicity, inputs=input_text, outputs=[classify_toxicity_output_toxic])
        with gr.Column():
            explain_toxicity_button = gr.Button("Nánari greining")
            explain_toxicity_output = gr.HTML()
            explain_toxicity_button.click(fn=explain_toxicity, inputs=input_text, outputs=explain_toxicity_output)
    with gr.Row():
        with gr.Column():
            classify_emotion_button = gr.Button("Greina algengustu tilfinningarnar í textanum")
            emotion_score1 = gr.Slider(minimum=0, maximum=1, label="", interactive=False)
            emotion_score2 = gr.Slider(minimum=0, maximum=1, label="", interactive=False)
            emotion_score3 = gr.Slider(minimum=0, maximum=1, label="", interactive=False)
            classify_emotion_button.click(fn=classify_emotion, inputs=input_text, outputs=[emotion_score1, emotion_score2, emotion_score3])
    with gr.Row():
        gr.HTML(
            """
            <div style='text-align: center; margin-top: 20px;'>
                <h2>ℹ️ Notkunarleiðbeiningar ℹ️</h2>
            </div>
            """
        )
    with gr.Row():
        gr.HTML(
            """
                <p>Þetta tól greinir textainntak. </p>
                <p>Vinstri hlið síðunnar sýnir, á skalanum 0 - 1, upp að hvaða marki textinn líkist ákveðnum flokkum á meðan hægri hluti síðunnar útskýrir hvernig tólið komst að þeirri niðurstöðu með því að sýna orðin sem höfðu áhrif á greininguna.</p>
                
                <p>Tólið leggur mat á lyndi texta sem getur verið neikvætt eða jákvætt. Það sýnir einnig þrjár megin tilfinningarnar sem fundust í textanum. Formlegheitagreining segir til um hvort texti talist formlegur eða óformlegur og að lokum getur tólið sagt til um það hvort óviðeigandi orð komi fram í textainntakinu. </p>
                <br>
                <p> Nánari greining fyrir lyndi, formlegheit og óviðeigandi orð virkar á eftirfarandi hátt: </p>
                <p>
                    <b>Lyndi: </b>Blá orð stuðla að neikvæðri greiningu. Appelsínugul orð stuðla að jákvæðri greiningu.<br>
                    <b>Formlegheit: </b>Blá orð stuðla að óformlegri greiningu. Appelsínugul orð stuðla að formlegri greiningu.<br>
                    <b>Óviðeigandi orð: </b>Orð sem eru rauð eru talin óviðeigandi.
                </p>
                <br>
                <p>
                    Orð sem eru ekki lituð (hvítt letur) eru ekki talin hafa mikil áhrif á greiningu textans. Því stærri og litríkari sem orðin eru, því meiri líkur eru á því að þau orð hafi haft áhrif á greininguna sem inntakið fékk. Þar sem öll líkön nema lyndisgreining eru þjálfuð á enskum gögnum, er endurgjöfin fyrir þau á þýddum texta.
                </p>
                <br>
                <p>
                    <b>Dæmi:</b> Ef fleiri en eitt orð eru lituð og í nánari greiningu á óviðeigandi orðum, þá mun það orð sem er stærst og litað skærasta rauða litnum vera það sem hefur mest áhrif á greininguna.
                </p>
            """
        )
    run_all_btn.click(
    fn=run_all,
    inputs=input_text,
    outputs=[
        classify_toxicity_output_toxic,
        classify_sentiment_output_negative,
        classify_sentiment_output_positive,
        emotion_score1,
        emotion_score2,
        emotion_score3,
        classify_formality_output_formal,
        classify_formality_output_informal,
        explain_toxicity_output,
        explain_sentiment_output,
        explain_formality_output
    ]
)

iface.launch(debug=True)
