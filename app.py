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
model_name_toxic = "unitary/toxic-bert"
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
explainer_emotion = shap.Explainer(pipeline_emotion, tokenizer_emotion)
explainer_formality = shap.Explainer(pipeline_formality, tokenizer_formality)

# Labels for each classifier 
class_labels_formality = ["formal", "informal"]
class_labels_sentiment = ["Negative", "Positive"]
class_labels_toxic = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
class_labels_emotion = [
    "disappointment", "sadness", "annoyance", "neutral", "disapproval",
    "realization", "nervousness", "approval", "joy", "anger", "embarrassment",
    "caring", "remorse", "disgust", "grief", "confusion", "relief", "desire",
    "admiration", "optimism", "fear", "love", "excitement", "curiosity",
    "amusement", "surprise", "gratitude", "pride"
]

# Define classification and explanation functions
def classify_toxicity(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    results = pipeline_toxic(translated_text)[0]  
    output = []
    
    toxic_result = next((item for item in results if item['label'] == 'toxic'), None)
    
    if toxic_result:
        toxicity_percentage = toxic_result['score'] * 100
        output.append(f"This sentence is {toxicity_percentage:.2f}% toxic.\n")
    else:
        output.append("This sentence is 0% toxic.\n")
        
    for result in results:
        label = result['label'].replace('_', ' ').capitalize()  
        score = result['score'] * 100
        output.append(f"{label}: {score:.2f}%")
    
    return "\n".join(output)


def explain_toxicity(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    shap_values = explainer_toxic([translated_text], fixed_context=1)

    toxic_index = class_labels_toxic.index("toxic")

    words_and_colors = []
    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, toxic_index]):
        escaped_word = html.escape(word)
        if shap_value > 0:
            color = '#b6ffb8'  # Light green for positive SHAP value
        elif shap_value < 0:
            color = '#ffccd1'  # Light red for negative SHAP value
        else:
            color = 'lightgrey'  # Neutral SHAP value
            words_and_colors.append(" ")
        words_and_colors.append(f"<span style='background-color:{color}; color:black;'>{escaped_word}</span>")
    #result_modified = "<strong>Modified:</strong> " + ''.join(words_and_colors)

    words_with_scores = []
    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, toxic_index]):
        escaped_word = html.escape(word)
        color = '#ffb6c1' if shap_value > 0 else '#808080'  # Light pink for positive SHAP value, grey for negative SHAP value
        words_with_scores.append(f"({round(shap_value, 4)}) <span style='color:{color};'>{escaped_word}</span>")
    result_with_scores = "<strong>With Scores:</strong> " + ' '.join(words_with_scores)

    #return result_modified + "<br><br>" + result_with_scores 
    return ''.join(words_and_colors)

### SENTIMENT
def classify_sentiment(sentence):
    try:
        results = pipeline_sentiment(sentence)[0] 
        output = []
        
        if results[0]['label'] == 'LABEL_0' and results[1]['label'] == 'LABEL_1':
            primary_sentiment = "Negative" if results[0]['score'] > results[1]['score'] else "Positive"
        elif results[0]['label'] == 'LABEL_1' and results[1]['label'] == 'LABEL_0':
            primary_sentiment = "Positive" if results[0]['score'] > results[1]['score'] else "Negative"
        else:
            primary_sentiment = "Unknown"
        
        output.append(f"Input classified as {primary_sentiment.lower()}.\n")
        
        for result in results:
            label = "Positive" if result['label'] == 'LABEL_1' else "Negative"
            score = result['score'] * 100
            output.append(f"{label}: {score:.2f}%")
        
        return "\n".join(output)
    except Exception as e:

        return f"Error processing the sentiment: {str(e)}"


def explain_sentiment(sentence):
    results = pipeline_sentiment(sentence)[0]  
    shap_values = explainer_sentiment([sentence], fixed_context=1) 

    positive_index = class_labels_sentiment.index("Positive")

    words_and_colors = []
    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, positive_index]):
        escaped_word = html.escape(word)
        if shap_value > 0:
            color = '#b6ffb8'  # Light green for positive SHAP value
        elif shap_value < 0:
            color = '#ffccd1'  # Light red for negative SHAP value
        else:
            color = 'lightgrey'  # Neutral SHAP value
            words_and_colors.append(" ")
        words_and_colors.append(f"<span style='background-color:{color}; color:black;'>{escaped_word}</span>")
    result_modified = "<strong>Modified:</strong> " + ''.join(words_and_colors)

    words_with_scores = []
    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, positive_index]):
        escaped_word = html.escape(word)
        color = '#ffb6c1' if shap_value > 0 else '#808080'  # Light pink for positive SHAP value, grey for negative SHAP value
        words_with_scores.append(f"({round(shap_value, 4)}) <span style='color:{color};'>{escaped_word}</span>")
    result_with_scores = "<strong>With Scores:</strong> " + ' '.join(words_with_scores)

    #return result_modified + "<br><br>" + result_with_scores 
    return ''.join(words_and_colors)

### EMOTION
def classify_emotion(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    results = pipeline_emotion(translated_text, top_k=len(class_labels_emotion))
    
    top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    
    output = ["The top 3 emotions detected in the text input are: \n"]
    for result in top_results:
        label = result['label'].capitalize() 
        score = result['score'] * 100 
        output.append(f"{label}: {score:.2f}%")
    
    return "\n".join(output)


def explain_emotion(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    shap_values = explainer_emotion([translated_text], fixed_context=1)

    top_emotions = ['joy', 'sadness', 'anger']  # Top 3 emotions to explain

    result_modified = "<strong>Modified for Top Emotions:</strong><br>"
    result_with_scores = "<strong>With Scores for Top Emotions:</strong><br>"

    for emotion in top_emotions:
        emotion_index = class_labels_emotion.index(emotion)

        words_and_colors = []
        for word, shap_value_array in zip(shap_values[0].data, shap_values[0].values):
            shap_value = shap_value_array[emotion_index]
            escaped_word = html.escape(word)
            if shap_value > 0:
                color = '#b6ffb8'  # Light green for positive SHAP value
            elif shap_value < 0:
                color = '#ffccd1'  # Light red for negative SHAP value
            else:
                color = 'lightgrey'  # Neutral SHAP value
                words_and_colors.append(" ")
            words_and_colors.append(f"<span style='background-color:{color}; color:black;'>{escaped_word}</span>")
        result_modified += f"<br>{emotion.capitalize()}: " + ''.join(words_and_colors)

        words_with_scores = []
        for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, emotion_index]):
            escaped_word = html.escape(word)
            color = '#ffb6c1' if shap_value > 0 else '#808080'  # Light pink for positive SHAP value, grey for negative SHAP value
            words_with_scores.append(f"({round(shap_value, 4)}) <span style='color:{color};'>{escaped_word}</span>")
        result_with_scores += f"<br>{emotion.capitalize()}: " + ' '.join(words_with_scores)

    return result_modified #+ "<br><br>" + result_with_scores 

### FORMALITY
def classify_formality(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    results = pipeline_formality(translated_text)[0] 
    output = []
    
    result_map = {result['label']: result['score'] * 100 for result in results}

    primary_formality = "formal" if result_map.get('formal', 0) > result_map.get('informal', 0) else "informal"
    
    output.append(f"Input classified as {primary_formality}.\n")
    
    for label in ['formal', 'informal']:
        score = result_map.get(label, 0) 
        output.append(f"{label.capitalize()}: {score:.2f}%")
    
    return "\n".join(output)

def explain_formality(sentence):
    translated_text = translator(sentence, max_length=512)[0]['translation_text']
    shap_values = explainer_formality([translated_text], fixed_context=1)

    formal_index = class_labels_formality.index("formal") 

    words_and_colors = []
    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, formal_index]):
        escaped_word = html.escape(word)
        # This model works the opposite way as the other, that's why the arrows are different.
        if shap_value < 0:
            color = '#b6ffb8'  # Light green for positive SHAP value
        elif shap_value > 0:
            color = '#ffccd1'  # Light red for negative SHAP value
        else:
            color = 'lightgrey'  # Neutral SHAP value
            words_and_colors.append(" ")
        words_and_colors.append(f"<span style='background-color:{color}; color:black;'>{escaped_word}</span>")
    #result_modified = "<strong>Modified:</strong> " + ''.join(words_and_colors)

    words_with_scores = []
    for word, shap_value in zip(shap_values[0].data, shap_values[0].values[:, formal_index]):
        escaped_word = html.escape(word)
        color = '#ffb6c1' if shap_value > 0 else '#808080'  # Light pink for positive SHAP value, grey for negative SHAP value
        words_with_scores.append(f"({round(shap_value, 4)}) <span style='color:{color};'>{escaped_word}</span>")
    result_with_scores = "<strong>With Scores:</strong> " + ' '.join(words_with_scores)

    #return result_modified + "<br><br>" + result_with_scores 
    return ''.join(words_and_colors)


# Create Gradio interface
with gr.Blocks() as iface:
    with gr.Row():
        gr.HTML(
            """
            <center>
                <img src='http://www.ru.is/media/HR_logo_vinstri_transparent.png' width='250' height='auto'>
            </center>
            """
        )
    with gr.Row():
        gr.HTML(
            """
            <div style='text-align: center; margin-top: 20px;'>
                <p>Welcome to the Text Analysis Tool! Please enter a sentence to analyze its toxicity, sentiment, emotion, or formality.</p>
                <p>You can also use the explanation buttons to see what influences the model's predictions.</p>
                <p>Note for the explainer: When something is green, that means it contributes to 1 labels, otherwise 0 labels.</p>
            </div>
            """
        )
    with gr.Row():
        input_text = gr.Textbox(lines=2, placeholder="Enter a sentence...")
    with gr.Row():
        with gr.Column():
            classify_toxicity_button = gr.Button("Calculate Toxicity")
            classify_toxicity_output = gr.Textbox()
            classify_toxicity_button.click(fn=classify_toxicity, inputs=input_text, outputs=classify_toxicity_output)
        with gr.Column():
            explain_toxicity_button = gr.Button("Explain Toxicity")
            explain_toxicity_output = gr.HTML()
            explain_toxicity_button.click(fn=explain_toxicity, inputs=input_text, outputs=explain_toxicity_output)
    with gr.Row():
        with gr.Column():
            classify_sentiment_button = gr.Button("Calculate Sentiment")
            classify_sentiment_output = gr.Textbox()
            classify_sentiment_button.click(fn=classify_sentiment, inputs=input_text, outputs=classify_sentiment_output)
        with gr.Column():
            explain_sentiment_button = gr.Button("Explain Sentiment")
            explain_sentiment_output = gr.HTML()
            explain_sentiment_button.click(fn=explain_sentiment, inputs=input_text, outputs=explain_sentiment_output)
    with gr.Row():
        with gr.Column():
            classify_emotion_button = gr.Button("Calculate Emotion")
            classify_emotion_output = gr.Textbox()
            classify_emotion_button.click(fn=classify_emotion, inputs=input_text, outputs=classify_emotion_output)
        with gr.Column():
            explain_emotion_button = gr.Button("Explain Emotion")
            explain_emotion_output = gr.HTML()
            explain_emotion_button.click(fn=explain_emotion, inputs=input_text, outputs=explain_emotion_output)
    with gr.Row():
        with gr.Column():
            classify_formality_button = gr.Button("Calculate Formality")
            classify_formality_output = gr.Textbox()
            classify_formality_button.click(fn=classify_formality, inputs=input_text, outputs=classify_formality_output)
        with gr.Column():
            explain_formality_button = gr.Button("Explain Formality")
            explain_formality_output = gr.HTML()
            explain_formality_button.click(fn=explain_formality, inputs=input_text, outputs=explain_formality_output)

iface.launch(debug=True)