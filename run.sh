#!/bin/bash

echo "Starting Histopathology Image Classifier WebApp..."

# Install dependencies if needed
echo "Installing dependencies..."
pip3 install flask flask-cors torch torchvision pillow opencv-python numpy

# Start backend
echo "Starting backend server..."
cd webapp/backend
python3 app.py &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend server..."
cd ../frontend
python3 -m http.server 8000 &
FRONTEND_PID=$!

echo "Backend running on http://localhost:5000"
echo "Frontend running on http://localhost:8000"
echo "Press Ctrl+C to stop both servers"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait