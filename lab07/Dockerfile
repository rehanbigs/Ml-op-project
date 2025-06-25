# Use AWS's base image that already understands Lambda
FROM public.ecr.aws/lambda/python:3.10

# 1. Install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r prod.txt

# 2. Copy code & weights
COPY app.py model.pt model_def.py ./

# 3. Tell Lambda what to run
CMD ["app.lambda_handler"]   # Mangum automatically creates this
