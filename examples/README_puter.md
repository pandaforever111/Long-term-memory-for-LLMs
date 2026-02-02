# Using Puter.js with GPT Memory Agent

This guide explains how to use Puter.js as an alternative to OpenAI with the GPT Memory Agent.

## What is Puter.js?

Puter.js is a JavaScript library that provides free, serverless access to OpenAI models (like GPT-4o, DALL-E) and other AI capabilities directly from frontend code without requiring API keys or backend infrastructure. It allows you to integrate AI capabilities into web applications without the need for an OpenAI API key or backend server.

## Setup

1. Configure the Memory Agent to use Puter.js:

   Edit your `config/default_config.yaml` file to set Puter.js as the AI provider:

   ```yaml
   # AI Provider setting
   ai_provider: "puter"  # Options: "openai", "puter"
   ```

2. Run the web demo:

   ```bash
   # Install Flask if you haven't already
   pip install flask

   # Run the Flask server
   python examples/serve_puter_demo.py
   ```

3. Open your browser and navigate to http://localhost:5000

## Web Demo Features

The Puter.js web demo provides a simple chat interface that demonstrates the Memory Agent's capabilities:

- Chat with the AI assistant
- Automatically extract and store memories from your messages
- View stored memories in the UI
- Experience memory-enhanced responses

## How It Works

The web demo uses Puter.js to access OpenAI models directly from the browser:

1. User messages are processed in the browser to extract potential memories
2. Memories are stored locally in the browser
3. When generating responses, relevant memories are retrieved and included in the prompt
4. Puter.js handles the communication with OpenAI's models without requiring an API key

## Python Integration

While the primary use case for Puter.js is in web applications, we've also provided a placeholder `PuterClient` implementation that can be used in Python code. This is primarily for demonstration purposes and would require actual integration with a web frontend in a real application.

You can see an example of how this would work in `examples/puter_usage.py`.

## Limitations

- The Puter.js integration works best in web applications
- The Python `PuterClient` implementation is a placeholder and would need to be connected to an actual web frontend in a real application
- Some advanced features may require additional configuration

## Additional Resources

- [Puter.js Documentation](https://js.puter.com/)
- [Puter.js GitHub Repository](https://github.com/HeyPuter/puter.js)