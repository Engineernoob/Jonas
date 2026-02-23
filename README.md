🧠 Jonas

Local-first AI assistant with a neural policy brain.

Jonas is not just a chatbot wrapper around an LLM.

He separates:

🗣 Language Model → speaks

🧠 Neural Policy Brain → decides how he speaks

🗂 Semantic Memory (SQLite + embeddings) → remembers context

📊 Online Feedback Loop → learns from your ratings

Fully local.
No cloud APIs.
Runs clean on Apple Silicon.

🔥 Philosophy

Most AI assistants are just prompt wrappers.

Jonas is different.

Instead of letting the LLM decide everything, Jonas uses a small PyTorch neural network to:

Adjust temperature dynamically

Control verbosity

Inject subtle humor

Decide how much memory to recall

Learn from /rate feedback

The LLM generates language.

The policy brain shapes behavior.

🏗 Architecture
User Input
    ↓
Embedding (local via Ollama)
    ↓
Neural Policy Brain (PyTorch MLP)
    ↓
Memory Recall (SQLite + cosine similarity)
    ↓
LLM (Ollama)
    ↓
Response
    ↓
Optional /rate feedback → trains policy
🎭 Personality

Jonas carries:

Tactical calm operator energy

Intelligent older-brother warmth

Quiet strategic executive clarity

Subtle, controlled wit

Grounded.
Culturally fluent.
Never theatrical.
Never dismissive.

⚙️ Setup (Mac / Apple Silicon Recommended)
1️⃣ Install Ollama
brew install ollama
ollama serve
2️⃣ Pull models

Recommended:

ollama pull llama3.1:8b
ollama pull nomic-embed-text

(You can use llama3.2:1b for lightweight testing.)

3️⃣ Python Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
4️⃣ Run Jonas
python -m src.main
💬 Commands
Command	Description
/quit	Exit
/reset	Clear memory
/history	Show recent turns
/rate 0.8	Train Jonas on last response
🧠 Neural Policy Brain

The policy network:

Small PyTorch MLP

Learns from scalar reward (0–1)

Saves weights to data/policy.pt

Adapts tone and structure over time

No reinforcement learning frameworks.
No heavy infrastructure.
Just incremental behavioral shaping.

🔒 Privacy

All inference is local via Ollama.

Memory stored in local SQLite database.

No external API calls.

No telemetry.

📁 Project Structure
jonas/
│
├── src/
│   ├── main.py
│   ├── policy_brain.py
│   ├── persona.py
│   ├── encoder.py
│   ├── memory.py
│   ├── ollama_client.py
│
├── data/
│   ├── chat.db
│   ├── policy.pt
│
├── requirements.txt
└── README.md
🚀 Roadmap

Planned improvements:

 Policy smoothing (reduce tone switching)

 Working-memory summarization

 Structured task mode

 Optional lightweight GUI

 Long-term preference tagging

 Better reward shaping

🧩 Why Jonas Exists

To experiment with:

Policy steering for LLMs

Local-first AI systems

Minimal RL-style feedback loops

Human-in-the-loop tone adaptation

Jonas is a research playground disguised as a personal assistant.

⚠️ Notes

Smaller models (1B) may exaggerate tone.

8B models handle nuance significantly better.

Delete data/policy.pt if you change embedding size.

🪶 License

MIT