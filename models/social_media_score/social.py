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

# Define columns and weights for each platform
COLUMNS_AND_WEIGHTS = {
    'twitter': {
        'columns': ['Tweets', 'Re-Tweets', 'Engagement', 'Bio', 'Followers', 'Following', 'Community'], 
        'weights': {'Tweets': 0.9, 'Re-Tweets': 0.8, 'Engagement': 0.8, 'Bio': 0.9, 'Followers': 0.6, 'Following': 0.6, 'Community': 0.7},
    },
    'linkedin': {
        'columns': ['Company', 'Position', 'Skills', 'Connections', 'Certification', 'Job_Title', 'QnA', 'Authority'], 
        'weights': {'Company': 0.7, 'Position': 0.5, 'Skills': 0.5, 'Connections': 0.8, 'Certification': 0.5, 'Job_Title': 0.7, 'QnA': 0.6, 'Authority': 0.4},
    },
    'instagram': {
        'columns': ['Story_Like', 'Post_Like', 'Post_Comment', 'Followers', 'Following', 'Follow_Request', 'Posts', 'Shop', 'Business'], 
        'weights': {'Story_Like': 0.4, 'Post_Like': 0.4, 'Post_Comment': 0.8, 'Followers': 0.6, 'Following': 0.6, 'Follow_Request': 0.3, 'Posts': 0.7, 'Shop': 0.5, 'Business': 0.5},
    },
}

# Function to normalize the total sentiment to the range 390-900
def normalize_to_credit_score(total_sentiment, max_value):
    normalized_value = total_sentiment / max_value
    scaled_value = 390 + (normalized_value * (900 - 390))
    return scaled_value

sentiment_mapping = {"Bad": 0, "Neutral": 1, "Good": 2}

# Define a route within the blueprint for the home page
@social_media_blueprint.route('/social_media_form')
def social_media_form():
    # Capture predicted value from query parameter, defaulting to 'Neutral' if not provided
    predicted_value = request.args.get('predicted_value', 'Neutral')
    mapped_value = sentiment_mapping.get(predicted_value, 1)
    return render_template('social_media_template.html', textbox_value=predicted_value)

# Define a route for processing sentiment analysis
@social_media_blueprint.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        if not all(['file1' in request.files, 'file2' in request.files, 'file3' in request.files]):
            return 'All three files are required', 400
        
        predicted_value = request.form.get('predicted_value', 'Neutral')
        mapped_value = sentiment_mapping.get(predicted_value, 1)
        files = {
            'twitter': request.files['file1'],
            'linkedin': request.files['file2'],
            'instagram': request.files['file3'],
        }
        total_sentiment = {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}
        
        for platform, file in files.items():
            if file.filename == '':
                return f'The {platform} file is empty', 400
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                platform_config = COLUMNS_AND_WEIGHTS.get(platform, {})
                platform_columns = platform_config.get('columns', [])
                weights = platform_config.get('weights', {})
                
                for column in platform_columns:
                    if column not in df.columns:
                        continue
                    
                    # Combine text from the column and get sentiment
                    combined_text = ' '.join(df[column].dropna().astype(str))
                    sentiment = get_sentiment(combined_text)
                    
                    # Apply weights and add to total sentiment
                    weight = weights.get(column, 1)  # Default weight is 1
                    total_sentiment['roberta_neg'] += sentiment['roberta_neg'] * weight
                    total_sentiment['roberta_neu'] += sentiment['roberta_neu'] * weight
                    total_sentiment['roberta_pos'] += sentiment['roberta_pos'] * weight
            else:
                return f'Invalid file format for {platform}. Please upload a CSV file.', 400
        
        # Normalize the sentiment scores
        total_score_sum = sum(total_sentiment.values())

        if total_score_sum == 0:
        # Handle zero division case
            total_sentiment = {key: 0 for key in total_sentiment}
        else:
            total_sentiment = {key: score / total_score_sum for key, score in total_sentiment.items()}

        max_value = 31.2
        weight_sentiment = 0.6
        weight_traditional = 0.4
 

        # Determine sentiment label and credit score
        if total_sentiment['roberta_pos'] > max(total_sentiment['roberta_neg'], total_sentiment['roberta_neu']):
            label = 'POSITIVE'
            score = total_sentiment['roberta_pos']
            credit_score = (
                weight_sentiment * score + 
                weight_traditional * mapped_value
            )

        elif total_sentiment['roberta_neg'] > max(total_sentiment['roberta_pos'], total_sentiment['roberta_neu']):
            label = 'NEGATIVE'
            score = total_sentiment['roberta_neg']
            credit_score = (
                weight_sentiment * score + 
                weight_traditional * mapped_value
            )

        else:
            label = 'NEUTRAL'
            score = total_sentiment['roberta_neu']
            credit_score = (
                weight_sentiment * score + 
                weight_traditional * mapped_value
            )

        
        final_credit_score = normalize_to_credit_score(credit_score, max_value)

        result = {
            'label': label,
            'score': score,
            'credit_score': int(final_credit_score),
        }

        return render_template('social_media_template.html', result=result)

# Register the blueprint with the app
@social_media_blueprint.route('/report', methods=['GET'])
def report():
    name = request.args.get('name', 'N/A')
    email = request.args.get('email', 'N/A')
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