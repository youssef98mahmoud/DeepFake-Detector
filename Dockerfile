FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime@sha256:8d409f72f99e5968b5c4c9396a21f4b723982cfdf2c1a5b9cc045c5d0a7345a1

# Update image
RUN : \
    && apt-get -y update -qq \
    && apt-get -y install wget debsecan \
    && apt-get install --no-install-recommends -y $(debsecan --suite bookworm --format packages --only-fixed) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /deepfake/models /safe_data /safe_outputs /scratch

# Install dependencies
COPY requirements.txt /scratch
RUN pip install --no-cache-dir -r /scratch/requirements.txt
RUN rm /scratch/requirements.txt

# Copy code
COPY deepfake_detection.py /deepfake/
COPY deepfake_detection/ /deepfake/deepfake_detection/

# Download models
RUN wget -P /deepfake/models https://download.pytorch.org/models/resnet18-f37072fd.pth

# Set torch path to local models
ENV TORCH_HOME="/deepfake/models"

# Run code
CMD ["python", "/deepfake/deepfake_detection.py", "-c", "/safe_data/deepfake/Deepfake.conf.yaml", "-o"]
