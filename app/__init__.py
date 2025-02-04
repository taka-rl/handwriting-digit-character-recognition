from flask import Flask
from app.models import load_model

def create_app():
    # Create Flask app instance
    app = Flask(__name__, template_folder='./templates', static_folder='./static')

    # Preload models at startup
    load_model("model_digit")
    load_model("model_character")

    # Import blueprints (routes)
    from app.routes.index import index_bp
    from app.routes.canvas import canvas_bp
    from app.routes.import_file import import_file_bp

    # Register blueprints
    app.register_blueprint(index_bp)
    app.register_blueprint(canvas_bp)
    app.register_blueprint(import_file_bp)

    return app
