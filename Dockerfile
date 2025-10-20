# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required by the project
# This replaces the packages.txt and apt-get install step
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the Streamlit configuration file directly, replacing setup.sh
RUN mkdir -p .streamlit/ && \
    echo "\
[theme]\n\
primaryColor=\\\"#00C4FF\\\"\n\
backgroundColor=\\\"#0E1117\\\"\n\
secondaryBackgroundColor=\\\"#262730\\\"\n\
textColor=\\\"#FAFAFA\\\"\n\
font=\\\"sans serif\\\"\n\
\n\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
" > .streamlit/config.toml

# Expose the port that Streamlit runs on
EXPOSE 8501

# The command to run the Streamlit application
CMD ["streamlit", "run", "Home.py"]
