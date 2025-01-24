from flask import Flask, render_template, redirect, url_for
from models.financial_score.app1 import financial_score_blueprint
from models.social_media_score.app2 import social_media_blueprint

app = Flask(__name__, template_folder='templates', static_folder='static')

# Register blueprints for financial score and social media
app.register_blueprint(financial_score_blueprint, url_prefix='/financial_score', template_folder='models/financial_score/templates')
app.register_blueprint(social_media_blueprint, url_prefix='/social_media_score', template_folder='models/social_media_score/templates')

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Financial form route
@app.route('/financial_form')
def financial_form():
    return render_template('financial_template.html')

# Social media form route
@app.route('/social_media_form')
def social_media_form():
    return render_template('social_media_template.html')

if __name__ == '__main__':
    app.run(debug=True)
