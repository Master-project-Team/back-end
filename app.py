from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime
from API import *

app = Flask(__name__)

# Connexion à la base de données SQLite
def get_db_connection():
    conn = sqlite3.connect('conversations.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialisation de la base de données
def init_db():
    with app.app_context():
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id INTEGER,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()
        db.close()

# Initialiser la base de données
init_db()

def generate_response(user_message, chat_llm_chain):
    # Utiliser le modèle LLM pour générer une réponse
    response = Reponse_Predicter(chat_llm_chain, user_message)
    # Extraire le texte de la réponse
    response_text = response['answer'] if 'answer' in response else "Erreur de génération de réponse"
    return response_text

# Route pour la page d'accueil
@app.route('/')
def index():
    conn = get_db_connection()
    conversations = conn.execute('SELECT DISTINCT conversation_id FROM conversations').fetchall()
    conn.close()
    return render_template('index.html', conversations=conversations)

# Route pour créer une nouvelle conversation
@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    conn = get_db_connection()
    cursor = conn.execute('SELECT MAX(conversation_id) FROM conversations')
    max_id = cursor.fetchone()[0]
    conn.close()

    if max_id:
        new_conversation_id = int(max_id) + 1
    else:
        new_conversation_id = 1

    return jsonify({'conversation_id': new_conversation_id})

# Route pour récupérer une conversation spécifique
@app.route('/conversation/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    conn = get_db_connection()
    messages = conn.execute('SELECT user_message, bot_response FROM conversations WHERE conversation_id = ?', (conversation_id,)).fetchall()
    conn.close()

    conversation = [{'user_message': message['user_message'], 'bot_response': message['bot_response']} for message in messages]
    
    return jsonify(conversation)

# Route pour créer une nouvelle conversation ou continuer une conversation existante
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    conversation_id = request.json.get('conversation_id')

    if not conversation_id:
        # Start a new conversation
        conn = get_db_connection()
        cursor = conn.execute('SELECT MAX(conversation_id) FROM conversations')
        max_id = cursor.fetchone()[0]
        conn.close()

        if max_id:
            conversation_id = int(max_id) + 1
        else:
            conversation_id = 1

    bot_response = generate_response(user_message, chat_llm_chain)

    conn = get_db_connection()
    conn.execute('INSERT INTO conversations (conversation_id, user_message, bot_response) VALUES (?, ?, ?)',
                 (conversation_id, user_message, bot_response))
    conn.commit()
    conn.close()
    
    return jsonify({'user_message': user_message, 'bot_response': bot_response, 'conversation_id': conversation_id})

# Route pour récupérer les liens des conversations
@app.route('/conversation_links', methods=['GET'])
def conversation_links():
    conn = get_db_connection()
    cursor = conn.execute('SELECT DISTINCT conversation_id FROM conversations')
    conversations = [row['conversation_id'] for row in cursor.fetchall()]
    conn.close()
    return jsonify(conversations)

if __name__ == '__main__':
    chat_llm_chain = run_pipeline_NContexte()
    app.run(debug=True)
