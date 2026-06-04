# LGTRx Studio-ready app

## Run in Studio

1. Create `.env` from `.env.example`.
2. Install dependencies:
   ```bash
   pip install -e .
   pip install -U "langgraph-cli[inmem]"
   ```
3. Start the local Agent Server:
   ```bash
   langgraph dev
   ```
4. Open the Studio URL printed by the CLI.

## Recommended Studio input

Use a payload like:

```json
{
  "input_messages": [
    {
      "role": "system",
      "content": "ID: 1, Region: X, Feedback: ..."
    },
    {
      "role": "user",
      "content": "Summarize the main concerns and recommendations."
    }
  ],
  "max_iterations": 3,
  "metadata": {
    "source": "studio"
  }
}
```

## Batch runner

You can still run the old batch-style workflow with:

```bash
python main.py
```
