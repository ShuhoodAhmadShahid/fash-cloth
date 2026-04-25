import os
import sys
import uvicorn

# This file now serves as a wrapper to run the FastAPI main app.
# This ensures backward compatibility with instructions that suggest running 'python app.py'.

def run_server():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change working directory to backend
    os.chdir(current_dir)
    
    # Add current dir to path
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    print("==========================================")
    print("AI Fashion Recommendation System")
    print("Starting FastAPI Backend...")
    print("==========================================")
    
    # Run uvicorn
    # Import app here to avoid early import issues
    from main import app
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
