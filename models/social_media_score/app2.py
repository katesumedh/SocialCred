from flask import Flask, render_template, request, Blueprint
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import random

# Initialize the Flask app
app = Flask(__name__)

# Load the pretrained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Define the blueprint
social_media_blueprint = Blueprint('social_media_score', __name__, template_folder='templates')

# Function to get sentiment using RoBERTa
def get_sentiment(text):
    chunk_size = 512
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    total_scores = {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}
    
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        encoded_text = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        output = model(**encoded_text)
        scores = softmax(output[0][0].detach().numpy())
        
        total_scores['roberta_neg'] += scores[0]
        total_scores['roberta_neu'] += scores[1]
        total_scores['roberta_pos'] += scores[2]
    
    total_score_sum = sum(total_scores.values())
    normalized_scores = {key: score / total_score_sum for key, score in total_scores.items()}
    
    return normalized_scores

# Define a route within the blueprint for the home page
sentiment_mapping = {"Bad": 0, "Neutral": 1, "Good": 2}

@social_media_blueprint.route('/social_media_form')

def social_media_form():
    # Capture predicted value from query parameter, defaulting to 'Neutral' if not provided
    predicted_value = request.args.get('predicted_value', 'Neutral')
    mapped_value = sentiment_mapping.get(predicted_value, 1) # Default to Neutral = 1 if no match
    return render_template('social_media_template.html', textbox_value=predicted_value)


# Define a route for processing sentiment analysis (for three CSV files)
@social_media_blueprint.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        if not all(['file1' in request.files, 'file2' in request.files, 'file3' in request.files]):
            return 'All three files are required', 400
        predicted_value = request.form.get('predicted_value', 'Neutral')
        mapped_value = sentiment_mapping.get(predicted_value, 1)
        files = [request.files['file1'], request.files['file2'], request.files['file3']]
        total_sentiment = {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}
        
        for file in files:
            if file.filename == '':
                return 'One or more files are empty', 400
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                combined_text = ' '.join(df.astype(str).apply(' '.join, axis=1))
                sentiment = get_sentiment(combined_text)
                total_sentiment['roberta_neg'] += sentiment['roberta_neg']
                total_sentiment['roberta_neu'] += sentiment['roberta_neu']
                total_sentiment['roberta_pos'] += sentiment['roberta_pos']
            else:
                return 'Invalid file format. Please upload CSV files only.', 400
        
        # Averaging the scores across all files
        total_sentiment = {key: score / len(files) for key, score in total_sentiment.items()}
        
        if total_sentiment['roberta_pos'] > max(total_sentiment['roberta_neg'], total_sentiment['roberta_neu']):
            label = 'POSITIVE'
            score = total_sentiment['roberta_pos']
            if mapped_value == 0:
                credit_score = random.randint(550, 700)
            elif mapped_value == 1:
                credit_score = random.randint(700, 800)
            else:
                credit_score = random.randint(800, 890)
        elif total_sentiment['roberta_neg'] > max(total_sentiment['roberta_pos'], total_sentiment['roberta_neu']):
            label = 'NEGATIVE'
            score = total_sentiment['roberta_neg']
            if mapped_value == 0:
                credit_score = random.randint(300, 400)
            elif mapped_value == 1:
                credit_score = random.randint(400, 500)
            else:
                credit_score = random.randint(500, 600)
        else:
            label = 'NEUTRAL'
            score = total_sentiment['roberta_neu']
            if mapped_value== 0:
                credit_score = random.randint(550, 650)
            elif mapped_value == 1:
                credit_score = random.randint(650, 700)
            else:
                credit_score = random.randint(700, 750)

        result = {
            'label': label,
            'score': score,
            'credit_score': credit_score,
        }
        
        return render_template('social_media_template.html', result=result)
    
@social_media_blueprint.route('/report', methods=['GET'])
def report():
    name = request.args.get('name', 'N/A')
    email = request.args.get('email', 'N/A')
    # predictedValue = request.args.get('predictedValue', 'N/A')  # Make sure 'predictedValue' is fetched
    credit_score = int(request.args.get('credit_score', 0))
    return render_template('report.html', name=name, email=email, credit_score=credit_score)
# Register the blueprint with the app
app = Flask(__name__)
# Register the blueprint with the app
app.register_blueprint(social_media_blueprint, url_prefix='/social_media_score')

# Define a route for the home page (optional, but it's where you start the app)
@app.route('/')
def home():
    return render_template('social_media_template.html')

if __name__ == "_main_":
    app.run(debug=True)



